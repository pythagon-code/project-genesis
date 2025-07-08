package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.neurons;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain.SimulatorConfig;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs.Graph;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs.GraphNode;
import jakarta.annotation.Nonnull;

public class RelayNeuronFactory implements Closeable {
    @JsonProperty("Levels") private final List<List<RelayNeuron>> levels;
    @JsonProperty("SimulatorConfig") public final SimulatorConfig simulatorConfig;

    @JsonCreator
    private RelayNeuronFactory(
        @JsonProperty("Levels") @Nonnull List<List<RelayNeuron>> neurons,
        @JsonProperty("SimulatorConfig") @Nonnull SimulatorConfig simulatorConfig
    ) {
        this.levels = neurons;
        this.simulatorConfig = simulatorConfig;
    }

    public RelayNeuronFactory(@Nonnull SimulatorConfig simulatorConfig) {
        levels = new ArrayList<>();
        this.simulatorConfig = simulatorConfig;
    }

    public void build() {
        levels.add(new ArrayList<>());
        levels.getFirst().add(createNeuronOfLevel(0));

        Map<String, Graph> tagToGraphMap = new TreeMap<>();
        for (Graph graph : simulatorConfig.graphList()) {
            tagToGraphMap.put(graph.tag, graph);
        }

        if (levels.isEmpty()) {
            for (int i = 1; i < simulatorConfig.modelConfig().levels().size(); i++) {
                levels.add(new ArrayList<>());
                Graph graph = tagToGraphMap.get(simulatorConfig.modelConfig().levels().get(i).graph());
                List<RelayNeuron> level = levels.getLast();
                for (RelayNeuron parent : levels.get(i - 1)) {
                    for (GraphNode node : graph) {
                        RelayNeuron neuron = createNeuronOfLevel(i);
                        parent.addChild(neuron);
                        level.add(neuron);
                    }

                    for (GraphNode node : graph) {
                        for (GraphNode neighbor : node.getAdjacentNodes()) {
                            level.get(node.id).addNeighbor(level.get(neighbor.id));
                        }
                    }
                }
            }
        }
    }

    private @Nonnull RelayNeuron createNeuronOfLevel(int level) {
        if (level == simulatorConfig.modelConfig().levels().size() - 1) {
            return new PrimitiveNeuron();
        } else {
            return new MetaNeuron(level);
        }
    }

    @Override
    public void close() {
        for (RelayNeuron neuron : levels.stream().flatMap(List::stream).toList()) {
            neuron.close();
        }
    }
}