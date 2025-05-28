package edu.illinois.abhayp4.projectgenesis.cerebrum.brain;

import java.io.*;
import java.util.Map;

import jakarta.annotation.Nonnull;
import org.yaml.snakeyaml.Yaml;

public record SimulatorSettings(
    @Nonnull Map<String, Object> systemObject,
    @Nonnull Map<String, Object> modelArchitectureObject,
    @Nonnull Map<String, Object> transformersObject,
    @Nonnull Map<String, Object> neuronTopologyObject,
    @Nonnull Map<String, Object> baseNeuronObject,
    @Nonnull Map<String, Object> graphStructuresObject,
    @Nonnull Map<String, Object> optimizationObject
) {
    public static @Nonnull SimulatorSettings loadFromConfig(String config) throws IOException {
        try (
            InputStream systemStream = getConfigStream(config, "system.yml");
            InputStream modelArchitectureStream = getConfigStream(config, "model_architecture.yml");
            InputStream transformersStream = getConfigStream(config, "transformers.yml");
            InputStream neuronTopologyStream = getConfigStream(config, "neuron_topology.yml");
            InputStream baseNeuronStream = getConfigStream(config, "base_neuron.yml");
            InputStream graphStructuresStream = getConfigStream(config, "graph_structures.yml");
            InputStream optimizationStream = getConfigStream(config, "optimization.yml");
        ) {
            return new SimulatorSettings(
                new Yaml(),
                systemStream,
                modelArchitectureStream,
                transformersStream,
                neuronTopologyStream,
                baseNeuronStream,
                graphStructuresStream,
                optimizationStream
            );
        }
    }

    private static InputStream getConfigStream(String config, String resourceName) {
        return SimulatorSettings.class.getResourceAsStream("/" + config + "/" + resourceName);
    }

    private SimulatorSettings(
        Yaml yaml,
        InputStream systemStream,
        InputStream modelArchitectureStream,
        InputStream transformersStream,
        InputStream neuronTopologyStream,
        InputStream baseNeuronStream,
        InputStream graphStructuresStream,
        InputStream optimizationStream
    ) {
        this(
            yaml.load(systemStream),
            yaml.load(modelArchitectureStream),
            yaml.load(transformersStream),
            yaml.load(neuronTopologyStream),
            yaml.load(baseNeuronStream),
            yaml.load(graphStructuresStream),
            yaml.load(optimizationStream)
        );
    }
}