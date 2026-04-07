using System.Text.Json.Serialization;

namespace MoeViz.Models;

public record MoeProfile
{
    [JsonPropertyName("model")] public string Model { get; init; } = "";
    [JsonPropertyName("prompt")] public string Prompt { get; init; } = "";
    [JsonPropertyName("timestamp")] public string Timestamp { get; init; } = "";
    [JsonPropertyName("context_size")] public int ContextSize { get; init; }
    [JsonPropertyName("temperature")] public float Temperature { get; init; }
    [JsonPropertyName("n_expert")] public int NExpert { get; init; }
    [JsonPropertyName("n_expert_used")] public int NExpertUsed { get; init; }
    [JsonPropertyName("n_prefill")] public int NPrefill { get; init; }
    [JsonPropertyName("n_generated")] public int NGenerated { get; init; }
    [JsonPropertyName("layers")] public int[] Layers { get; init; } = [];
    [JsonPropertyName("frames")] public Frame[] Frames { get; init; } = [];

    public int NLayers => Layers.Length;
    public int NFrames => Frames.Length;
}

public record Frame
{
    [JsonPropertyName("gen")] public bool Gen { get; init; }
    [JsonPropertyName("token")] public string Token { get; init; } = "";
    [JsonPropertyName("sel")] public int[][] Sel { get; init; } = [];
    [JsonPropertyName("top")] public float[][][] Top { get; init; } = [];
    [JsonPropertyName("cands")] public Candidate[] Cands { get; init; } = [];
}

public record Candidate
{
    [JsonPropertyName("t")] public string T { get; init; } = "";
    [JsonPropertyName("p")] public float P { get; init; }
}
