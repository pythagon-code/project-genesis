package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import jakarta.annotation.Nonnull;

import java.util.List;
import java.util.Map;

public final class GraphFactory {
    @SuppressWarnings("unchecked")
    public @Nonnull Graph createGraph(@Nonnull Map<String, Object> object) {
        String tag = (String) object.get("tag");
        String graphType = object.get("graph_type").toString();
        Map<String, Object> graphOptions = (Map<String, Object>) object.get("graph_options");

        switch (tag) {
            case "perfect-tree":
                return new PerfectTreeGraph(
                    tag,
                    (int) graphOptions.get("m"),
                    (int) graphOptions.get("height")
                );
            case "fractal-tree":
                return new FractalTreeGraph(
                    tag,
                    (int) graphOptions.get("m"),
                    (int) graphOptions.get("height")
                );
            case "complete":
                return new CompleteGraph(
                    tag,
                    (int) graphOptions.get("n")
                );
            case "walk":
                return new WalkGraph(
                    tag,
                    (int) graphOptions.get("n")
                );
            case "cycle":
                return new CycleGraph(
                    tag,
                    (int) graphOptions.get("n")
                );
            case "wheel":
                return new WheelGraph(
                    tag,
                    (int) graphOptions.get("n")
                );
            case "custom":
                return new CustomGraph(
                    tag,
                    (int) graphOptions.get("n_vertices"),
                    (List<List<Integer>>) graphOptions.get("edges")
                );
            default:
                throw new IllegalArgumentException("Unknown graph type: " + graphType);
        }
    }
}
