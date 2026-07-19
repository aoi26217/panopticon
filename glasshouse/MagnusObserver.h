// ============================================================================
// Project Panopticon — Phase 6: The Glasshouse (UE5 client half)
// AMagnusObserver.h
//
// Subscriber-only LiveKit participant living inside Unreal Engine 5.
// Receives:
//   * the 20 Hz binary state stream on data-channel topic
//     "panopticon.state"   → drives pawn transforms (Hermite-interpolated)
//   * per-agent Opus audio tracks named "agent-{id}" → routed into
//     spatialized audio components at each pawn's live world position.
//
// WIRE CONTRACT — must remain byte-identical to broadcaster.py:
//   HEADER  : uint32 tick, uint16 agent_count            (6 bytes, LE)
//   AGENT   : char[8] id, float x,y,vx,vy, uint8 flags   (25 bytes, LE)
// The static_asserts below are the compile-time handshake with the Python
// side; if either half changes, the build breaks before the demo does.
//
// Interpolation: the sim ticks every 50 ms; the renderer runs at 60–120 fps.
// Each pawn keeps the two most recent snapshots and evaluates a cubic
// Hermite spline in Tick(): positions are the endpoints, the transmitted
// velocities are the tangents (that is exactly why vx/vy ride the wire —
// they are free Hermite tangents, not decoration). Result: fluid motion
// through single-frame loss on the unreliable channel.
// ============================================================================

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/AudioComponent.h"
// LiveKit C++ SDK (github.com/livekit/client-sdk-cpp). API surface is still
// settling; the integration seams are isolated in MagnusLiveKitBridge so SDK
// version churn touches exactly one .cpp.
#include "livekit/room.h"

#include "MagnusObserver.generated.h"

// ---------------------------------------------------------------------------
// Wire structs — packed, little-endian, mirrored from broadcaster.py
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct FGlasshouseHeader
{
    uint32 TickCount;
    uint16 AgentCount;
};

struct FGlasshouseAgentRecord
{
    char   AgentId[8];      // NUL-padded compacted id, e.g. "agent007"
    float  X;
    float  Y;
    float  Vx;
    float  Vy;
    uint8  Flags;           // bit0 speaking, bit1 degraded
};
#pragma pack(pop)

static_assert(sizeof(FGlasshouseHeader) == 6,       "header drifted from wire");
static_assert(sizeof(FGlasshouseAgentRecord) == 25, "record drifted from wire");

constexpr uint8 GLASSHOUSE_FLAG_SPEAKING = 0x01;
constexpr uint8 GLASSHOUSE_FLAG_DEGRADED = 0x02;
constexpr float GLASSHOUSE_TICK_DT       = 0.05f;   // 20 Hz sim period
constexpr float GLASSHOUSE_WORLD_TO_UU   = 100.0f;  // 1 sim unit = 1 m = 100 uu

// Two-snapshot buffer per agent: the Hermite endpoints.
struct FAgentSnapshot
{
    FVector2f Pos       = FVector2f::ZeroVector;
    FVector2f Vel       = FVector2f::ZeroVector;
    uint32    Tick      = 0;
    double    RecvTime  = 0.0;    // local clock — drives interpolation alpha
    uint8     Flags     = 0;
};

struct FAgentChannel
{
    FAgentSnapshot Prev;
    FAgentSnapshot Latest;
    TWeakObjectPtr<class APanopticonPawn> Pawn;
    TWeakObjectPtr<UAudioComponent>       Voice;   // spatialized agent audio
};

UCLASS()
class GLASSHOUSE_API AMagnusObserver : public AActor
{
    GENERATED_BODY()

public:
    AMagnusObserver();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;
    virtual void Tick(float DeltaTime) override;

    UPROPERTY(EditAnywhere, Category = "Glasshouse|Connection")
    FString LiveKitUrl = TEXT("wss://<runpod-host>:7880");

    UPROPERTY(EditAnywhere, Category = "Glasshouse|Connection")
    FString RoomName = TEXT("panopticon");

    UPROPERTY(EditAnywhere, Category = "Glasshouse|Connection")
    FString SubscriberToken;      // minted server-side; subscribe-only grant

    UPROPERTY(EditAnywhere, Category = "Glasshouse|Rendering")
    TSubclassOf<class APanopticonPawn> PawnClass;

private:
    // ---- LiveKit callbacks (arrive on SDK threads — marshal to game thread)
    void OnDataPacket(const uint8* Data, int32 Size, const FString& Topic);
    void OnAudioTrackSubscribed(const FString& TrackName,
                                livekit::AudioTrack* Track);

    // ---- Parsing & state
    bool ParseStatePacket(const uint8* Data, int32 Size);   // bit-exact decode
    FAgentChannel& ChannelFor(const FString& AgentId);

    // ---- Interpolation (runs in Tick on the game thread)
    FVector EvalHermite(const FAgentChannel& Ch, double Now) const;

    TMap<FString, FAgentChannel> Channels;
    FCriticalSection SnapshotLock;    // SDK thread writes, game thread reads
    TUniquePtr<class FMagnusLiveKitBridge> Bridge;   // all SDK seams live here

    uint32 LastAppliedTick = 0;       // stale/duplicate frame rejection
};
