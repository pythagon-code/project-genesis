package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs;

import jakarta.annotation.Nonnull;

import java.util.List;
import java.util.Map;

public final class GraphFactory {
    @SuppressWarnings("unchecked")
    public static @Nonnull Graph createGraph(@Nonnull Map<String, Object> object) {
        String tag = (String) object.get("tag");
        String graphType = object.get("graph_type").toString();
        Map<String, Object> graphOptions = (Map<String, Object>) object.get("graph_options");

        return switch (graphType) {
            case "perfect-tree" -> new PerfectTreeGraph(
                tag,
                (int) graphOptions.get("m"),
                (int) graphOptions.get("height")
            );
            case "fractal-tree" -> new FractalTreeGraph(
                tag,
                (int) graphOptions.get("m"),
                (int) graphOptions.get("height")
            );
            case "complete" -> new CompleteGraph(
                tag,
                (int) graphOptions.get("n")
            );
            case "walk" -> new WalkGraph(
                tag,
                (int) graphOptions.get("n")
            );
            case "cycle" -> new CycleGraph(
                tag,
                (int) graphOptions.get("n")
            );
            case "wheel" -> new WheelGraph(
                tag,
                (int) graphOptions.get("n")
            );
            case "custom" -> new CustomGraph(
                tag,
                (int) graphOptions.get("vertex_count"),
                (List<List<Integer>>) graphOptions.get("edges")
            );
            default -> throw new IllegalArgumentException("Unknown graph type: " + graphType);
        };
    }
}