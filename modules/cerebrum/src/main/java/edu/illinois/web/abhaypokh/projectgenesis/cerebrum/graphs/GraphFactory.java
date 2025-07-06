package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs;

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain.SimulatorConfig;
import jakarta.annotation.Nonnull;

import java.util.List;
import java.util.Map;

public final class GraphFactory {
    @SuppressWarnings("unchecked")
    public @Nonnull Graph createGraph(@Nonnull SimulatorConfig.GraphsConfig.GraphStructure graphStructure) {
        return switch (graphStructure.graphType()) {
            case "perfect-tree" -> new PerfectTreeGraph(
                graphStructure.tag(),
                (int) graphStructure.graphOptions().get("m"),
                (int) graphStructure.graphOptions().get("height")
            );
            case "fractal-tree" -> new FractalTreeGraph(
                graphStructure.tag(),
                (int) graphStructure.graphOptions().get("m"),
                (int) graphStructure.graphOptions().get("height")
            );
            case "complete" -> new CompleteGraph(
                graphStructure.tag(),
                (int) graphStructure.graphOptions().get("n")
            );
            case "walk" -> new WalkGraph(
                graphStructure.tag(),
                (int) graphStructure.graphOptions().get("n")
            );
            case "cycle" -> new CycleGraph(
                graphStructure.tag(),
                (int) graphStructure.graphOptions().get("n")
            );
            case "wheel" -> new WheelGraph(
                graphStructure.tag(),
                (int) graphStructure.graphOptions().get("n")
            );
            case "custom" -> new CustomGraph(
                graphStructure.tag(),
                (int) graphStructure.graphOptions().get("vertex_count"),
                (List<List<Integer>>) graphStructure.graphOptions().get("edges")
            );
            default -> throw new IllegalArgumentException("Unknown graph type: " + graphStructure.graphType());
        };
    }
}
