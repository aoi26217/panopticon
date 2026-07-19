// ============================================================================
// AMagnusObserver.cpp — Glasshouse client implementation
//
// The decode + interpolation + audio-mapping logic below is complete and
// engine-version-stable. The LiveKit SDK connection seams are isolated in
// FMagnusLiveKitBridge at the bottom; those calls track the C++ SDK's
// current surface and are the ONLY code expected to churn between SDK
// releases (marked SDK-SEAM).
// ============================================================================

#include "MagnusObserver.h"
#include "PanopticonPawn.h"      // simple skeletal pawn w/ speaking anim hooks
#include "Async/Async.h"

AMagnusObserver::AMagnusObserver()
{
    PrimaryActorTick.bCanEverTick = true;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void AMagnusObserver::BeginPlay()
{
    Super::BeginPlay();
    Bridge = MakeUnique<FMagnusLiveKitBridge>(*this);
    Bridge->Connect(LiveKitUrl, SubscriberToken);   // SDK-SEAM
}

void AMagnusObserver::EndPlay(const EEndPlayReason::Type Reason)
{
    if (Bridge) { Bridge->Disconnect(); }           // SDK-SEAM
    Super::EndPlay(Reason);
}

// ---------------------------------------------------------------------------
// Data path: SDK thread → parse → snapshot rotation
// ---------------------------------------------------------------------------

void AMagnusObserver::OnDataPacket(const uint8* Data, int32 Size,
                                   const FString& Topic)
{
    // Unreliable channel: packets may drop (fine — Hermite rides through)
    // or arrive out of order (rejected below via TickCount monotonicity).
    if (Topic != TEXT("panopticon.state")) { return; }
    ParseStatePacket(Data, Size);
}

bool AMagnusObserver::ParseStatePacket(const uint8* Data, int32 Size)
{
    if (Size < (int32)sizeof(FGlasshouseHeader)) { return false; }

    FGlasshouseHeader Header;
    FMemory::Memcpy(&Header, Data, sizeof(Header));   // packed LE == x86/ARM LE

    const int32 Expected =
        sizeof(FGlasshouseHeader) + Header.AgentCount * sizeof(FGlasshouseAgentRecord);
    if (Size != Expected) { return false; }           // torn frame: drop whole

    if (Header.TickCount <= LastAppliedTick &&
        LastAppliedTick - Header.TickCount < 0x7FFFFFFF)  // wrap tolerance
    {
        return false;                                 // stale/reordered frame
    }
    LastAppliedTick = Header.TickCount;

    const double Now = FPlatformTime::Seconds();
    const uint8* Cursor = Data + sizeof(FGlasshouseHeader);

    FScopeLock Lock(&SnapshotLock);
    for (uint16 i = 0; i < Header.AgentCount; ++i)
    {
        FGlasshouseAgentRecord Rec;
        FMemory::Memcpy(&Rec, Cursor, sizeof(Rec));
        Cursor += sizeof(Rec);

        // NUL-padded char[8] → FString (id registry mirrors broadcaster's)
        char IdBuf[9] = {0};
        FMemory::Memcpy(IdBuf, Rec.AgentId, 8);
        FAgentChannel& Ch = ChannelFor(FString(ANSI_TO_TCHAR(IdBuf)));

        Ch.Prev   = Ch.Latest;                        // rotate snapshots
        Ch.Latest = FAgentSnapshot{
            FVector2f(Rec.X, Rec.Y),
            FVector2f(Rec.Vx, Rec.Vy),
            Header.TickCount, Now, Rec.Flags };
    }
    return true;
}

FAgentChannel& AMagnusObserver::ChannelFor(const FString& AgentId)
{
    FAgentChannel* Ch = Channels.Find(AgentId);
    if (Ch) { return *Ch; }
    FAgentChannel& NewCh = Channels.Add(AgentId);
    // Pawn spawn must happen on the game thread; data arrives on SDK threads.
    AsyncTask(ENamedThreads::GameThread, [this, AgentId]()
    {
        if (FAgentChannel* C = Channels.Find(AgentId); C && !C->Pawn.IsValid())
        {
            C->Pawn = GetWorld()->SpawnActor<APanopticonPawn>(PawnClass);
            C->Pawn->SetAgentLabel(AgentId);
        }
    });
    return NewCh;
}

// ---------------------------------------------------------------------------
// Render path: 60–120 fps Hermite interpolation over 20 Hz snapshots
// ---------------------------------------------------------------------------

FVector AMagnusObserver::EvalHermite(const FAgentChannel& Ch, double Now) const
{
    // Alpha over the fixed sim period, referenced to receive time. Clamp to
    // ~1.5 so a lost frame extrapolates briefly along the velocity tangent
    // instead of freezing, then the next frame snaps the spline back.
    const float Alpha = FMath::Clamp(
        (float)((Now - Ch.Latest.RecvTime) / GLASSHOUSE_TICK_DT), 0.f, 1.5f);

    const FVector2f& P0 = Ch.Prev.Pos;
    const FVector2f& P1 = Ch.Latest.Pos;
    // Tangents scaled to the interval: velocity (units/s) × dt (s/interval).
    const FVector2f  M0 = Ch.Prev.Vel   * GLASSHOUSE_TICK_DT;
    const FVector2f  M1 = Ch.Latest.Vel * GLASSHOUSE_TICK_DT;

    const float T = Alpha, T2 = T * T, T3 = T2 * T;
    const FVector2f Pos =
        P0 * ( 2*T3 - 3*T2 + 1) +
        M0 * (   T3 - 2*T2 + T) +
        P1 * (-2*T3 + 3*T2    ) +
        M1 * (   T3 -   T2    );

    return FVector(Pos.X * GLASSHOUSE_WORLD_TO_UU,
                   Pos.Y * GLASSHOUSE_WORLD_TO_UU, 0.f);
}

void AMagnusObserver::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    const double Now = FPlatformTime::Seconds();

    FScopeLock Lock(&SnapshotLock);
    for (auto& Pair : Channels)
    {
        FAgentChannel& Ch = Pair.Value;
        if (!Ch.Pawn.IsValid()) { continue; }

        Ch.Pawn->SetActorLocation(EvalHermite(Ch, Now));
        Ch.Pawn->SetSpeaking(Ch.Latest.Flags & GLASSHOUSE_FLAG_SPEAKING);
        Ch.Pawn->SetDegraded(Ch.Latest.Flags & GLASSHOUSE_FLAG_DEGRADED);

        // Spatial audio follows the pawn: the attached component inherits
        // the interpolated transform every frame — sub-frame positional
        // accuracy for the F5-TTS voice.
        if (Ch.Voice.IsValid())
        {
            Ch.Voice->SetWorldLocation(Ch.Pawn->GetActorLocation());
        }
    }
}

// ---------------------------------------------------------------------------
// Audio path: LiveKit track "agent-{id}" → spatialized UAudioComponent
// ---------------------------------------------------------------------------

void AMagnusObserver::OnAudioTrackSubscribed(const FString& TrackName,
                                             livekit::AudioTrack* Track)
{
    // Track naming contract from media.py: "agent-{agent_id}".
    FString AgentId;
    if (!TrackName.Split(TEXT("agent-"), nullptr, &AgentId)) { return; }

    AsyncTask(ENamedThreads::GameThread, [this, AgentId, Track]()
    {
        FAgentChannel& Ch = ChannelFor(AgentId);
        UAudioComponent* Voice = NewObject<UAudioComponent>(this);
        Voice->bAllowSpatialization = true;
        Voice->bAutoActivate = true;
        Voice->AttenuationSettings = nullptr;   // project default attenuation
        Voice->RegisterComponent();
        // SDK-SEAM: bind the LiveKit audio sink to a USoundWaveProcedural
        // feeding this component (48 kHz frames from the Opus decoder).
        Bridge->BindTrackToComponent(Track, Voice);
        Ch.Voice = Voice;
        // NOTE (viseme contract): lip-sync is NOT parsed from any server
        // data — the WASM/ONNX viseme model taps this component's audio
        // buffer locally, exactly as specified since Phase 2.
    });
}

// ============================================================================
// FMagnusLiveKitBridge — every SDK-version-sensitive call, in one place.
// ============================================================================
//
// class FMagnusLiveKitBridge {
//   void Connect(url, token):
//     Room = livekit::Room::Create();
//     Room->AddListener(this);                       // SDK-SEAM
//     Room->Connect(TCHAR_TO_UTF8(*url), TCHAR_TO_UTF8(*token),
//                   { .auto_subscribe = true });     // SDK-SEAM
//   listener overrides:
//     OnDataReceived(payload, size, participant, topic)
//         → Owner.OnDataPacket(payload, size, topic)
//     OnTrackSubscribed(track, publication, participant)
//         → if (track->kind == Audio)
//               Owner.OnAudioTrackSubscribed(publication->name, track)
//   void BindTrackToComponent(track, comp):
//     audio_stream = livekit::AudioStream::Create(track, 48000, 1);
//     pump frames → USoundWaveProcedural::QueueAudio  // SDK-SEAM
// };
// ============================================================================
