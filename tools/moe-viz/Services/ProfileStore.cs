using MoeViz.Models;

namespace MoeViz.Services;

public class ProfileStore
{
    public List<MoeProfile> Profiles { get; } = new();
    public MoeProfile? Current { get; set; }

    public event Action? OnChange;

    public void Add(MoeProfile profile)
    {
        Profiles.Add(profile);
        Current = profile;
        OnChange?.Invoke();
    }
}
