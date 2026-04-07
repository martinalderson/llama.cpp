using MoeViz.Models;

namespace MoeViz.Services;

public static class StatsCalculator
{
    public static double Gini(int[] counts, long total)
    {
        if (total == 0 || counts.Length == 0) return 0;
        var sorted = counts.OrderBy(c => c).ToArray();
        int n = sorted.Length;
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += (2.0 * (i + 1) - n - 1.0) * sorted[i];
        return sum / (n * (double)total);
    }

    public static double AvgGini(MoeProfile profile, int upToFrame)
    {
        int nLayers = profile.NLayers;
        int nExpert = profile.NExpert;
        double giniSum = 0;

        for (int li = 0; li < nLayers; li++)
        {
            var counts = new int[nExpert];
            long total = 0;
            for (int fi = 0; fi <= upToFrame; fi++)
            {
                if (!profile.Frames[fi].Gen) continue;
                var sel = profile.Frames[fi].Sel;
                if (li >= sel.Length) continue;
                foreach (var eid in sel[li])
                {
                    if (eid >= 0 && eid < nExpert) { counts[eid]++; total++; }
                }
            }
            giniSum += Gini(counts, total);
        }
        return nLayers > 0 ? giniSum / nLayers : 0;
    }

    public static double AvgOverlap(MoeProfile profile, int frame)
    {
        if (frame <= 0) return 0;
        var prev = profile.Frames[frame - 1];
        var curr = profile.Frames[frame];
        if (!curr.Gen || !prev.Gen) return 0;

        int nLayers = profile.NLayers;
        int totalOverlap = 0;
        for (int li = 0; li < nLayers; li++)
        {
            if (li >= prev.Sel.Length || li >= curr.Sel.Length) continue;
            var prevSet = new HashSet<int>(prev.Sel[li]);
            foreach (var eid in curr.Sel[li])
                if (prevSet.Contains(eid)) totalOverlap++;
        }
        return nLayers > 0 ? (double)totalOverlap / nLayers : 0;
    }

    public static (int total, int recent) UniqueExperts(MoeProfile profile, int upToFrame, int window = 64)
    {
        int nLayers = profile.NLayers;
        int nExpert = profile.NExpert;
        int totalUniq = 0, recentUniq = 0;

        for (int li = 0; li < nLayers; li++)
        {
            var seenAll = new HashSet<int>();
            var seenRecent = new HashSet<int>();
            int start = Math.Max(0, upToFrame - window + 1);
            for (int fi = 0; fi <= upToFrame; fi++)
            {
                if (!profile.Frames[fi].Gen) continue;
                var sel = profile.Frames[fi].Sel;
                if (li >= sel.Length) continue;
                foreach (var eid in sel[li])
                {
                    if (eid >= 0 && eid < nExpert)
                    {
                        seenAll.Add(eid);
                        if (fi >= start) seenRecent.Add(eid);
                    }
                }
            }
            totalUniq += seenAll.Count;
            recentUniq += seenRecent.Count;
        }
        return (totalUniq, recentUniq);
    }

    public static (int expertId, int count, double pct)[] HottestPerLayer(MoeProfile profile, int upToFrame)
    {
        int nLayers = profile.NLayers;
        int nExpert = profile.NExpert;
        var result = new (int, int, double)[nLayers];
        int decTokens = 0;

        var counts = new int[nLayers][];
        for (int li = 0; li < nLayers; li++) counts[li] = new int[nExpert];

        for (int fi = 0; fi <= upToFrame; fi++)
        {
            if (!profile.Frames[fi].Gen) continue;
            decTokens++;
            for (int li = 0; li < nLayers; li++)
            {
                var sel = profile.Frames[fi].Sel;
                if (li >= sel.Length) continue;
                foreach (var eid in sel[li])
                    if (eid >= 0 && eid < nExpert) counts[li][eid]++;
            }
        }

        for (int li = 0; li < nLayers; li++)
        {
            int topE = 0, topC = 0;
            for (int ei = 0; ei < nExpert; ei++)
                if (counts[li][ei] > topC) { topC = counts[li][ei]; topE = ei; }
            result[li] = (topE, topC, decTokens > 0 ? 100.0 * topC / decTokens : 0);
        }
        return result;
    }

    public static int[,] CumulativeCounts(MoeProfile profile, int upToFrame)
    {
        int nLayers = profile.NLayers;
        int nExpert = profile.NExpert;
        var counts = new int[nLayers, nExpert];

        for (int fi = 0; fi <= upToFrame; fi++)
        {
            if (!profile.Frames[fi].Gen) continue;
            for (int li = 0; li < nLayers; li++)
            {
                var sel = profile.Frames[fi].Sel;
                if (li >= sel.Length) continue;
                foreach (var eid in sel[li])
                    if (eid >= 0 && eid < nExpert) counts[li, eid]++;
            }
        }
        return counts;
    }

    public static int CumulativeMax(MoeProfile profile)
    {
        var counts = CumulativeCounts(profile, profile.NFrames - 1);
        int max = 0;
        for (int li = 0; li < profile.NLayers; li++)
            for (int ei = 0; ei < profile.NExpert; ei++)
                if (counts[li, ei] > max) max = counts[li, ei];
        return max;
    }

    public static string EscapeToken(string token)
    {
        return token.Replace("\n", "\u21b5").Replace(" ", "\u00b7");
    }
}
