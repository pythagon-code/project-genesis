package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain;

import java.io.*;
import java.util.Map;

import jakarta.annotation.Nonnull;
import org.yaml.snakeyaml.Yaml;

public record SimulatorSettings(
    @Nonnull Map<String, Object> systemMap,
    @Nonnull Map<String, Object> graphsMap,
    @Nonnull Map<String, Object> modelMap,
    @Nonnull Map<String, Object> optimizationMap
) {
    public static @Nonnull SimulatorSettings loadFromConfig(@Nonnull String config) throws IOException {
        try (
            InputStream systemStream = getConfigStream(config, "system.yaml");
            InputStream graphsStream = getConfigStream(config, "graphs.yaml");
            InputStream modelStream = getConfigStream(config, "model.yaml");
            InputStream optimizationStream = getConfigStream(config, "optimization.yaml");
        ) {
            return new SimulatorSettings(
                new Yaml(),
                systemStream,
                graphsStream,
                modelStream,
                optimizationStream
            );
        }
    }

    private static InputStream getConfigStream(@Nonnull String config, @Nonnull String resourceName) {
        return SimulatorSettings.class.getResourceAsStream("/configs/" + config + "/" + resourceName);
    }

    private SimulatorSettings(
        @Nonnull Yaml yaml,
        @Nonnull InputStream systemStream,
        @Nonnull InputStream graphsStream,
        @Nonnull InputStream modelStream,
        @Nonnull InputStream optimizationStream
    ) {
        this(
            yaml.load(systemStream),
            yaml.load(graphsStream),
            yaml.load(modelStream),
            yaml.load(optimizationStream)
        );
    }
}