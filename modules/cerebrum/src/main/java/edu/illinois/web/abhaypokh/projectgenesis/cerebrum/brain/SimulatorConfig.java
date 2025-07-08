package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain;

import com.fasterxml.jackson.annotation.JsonProperty;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs.Graph;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs.GraphFactory;
import jakarta.annotation.Nonnull;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

public record  SimulatorConfig(
    @JsonProperty("SystemConfig") @Nonnull SystemConfig systemConfig,
    @JsonProperty("GraphsConfig") @Nonnull List<Graph> graphList,
    @JsonProperty("ModelConfig") @Nonnull ModelConfig modelConfig,
    @JsonProperty("OptimizationConfig") @Nonnull OptimizationConfig optimizationConfig
) {
    public SimulatorConfig(@Nonnull SimulatorSettings settings) {
        this(
            settings.systemMap(),
            settings.graphsMap(),
            settings.modelMap(),
            settings.optimizationMap()
        );
    }

    @SuppressWarnings("unchecked")
    private SimulatorConfig(
        @Nonnull Map<String, Object> systemMap,
        @Nonnull Map<String, Object> graphsMap,
        @Nonnull Map<String, Object> modelMap,
        @Nonnull Map<String, Object> optimizationMap
    ) {
        this(
            new SystemConfig((Map<String, Object>) systemMap.get("system")),
            getNestedFieldAsList(GraphFactory::createGraph, graphsMap, "graphs"),
            new ModelConfig((Map<String, Object>) modelMap.get("model")),
            new OptimizationConfig((Map<String, Object>) optimizationMap.get("optimization"))
        );
    }

    public record SystemConfig(
        @JsonProperty("PythonWorkerCount") int pythonWorkerCount,
        @JsonProperty("UseCuda") boolean useCuda,
        @JsonProperty("PreloadFrom") String preloadFrom,
        @JsonProperty("CheckpointFrequency") int checkpointFrequency

    ) {
        private SystemConfig(@Nonnull Map<String, Object> map) {
            this(
                getNestedField(map, "python_worker_count"),
                getNestedField(map, "use_cuda"),
                getNestedField(map, "preload_from"),
                getNestedField(map, "checkpoint_frequency")
            );
        }
    }

    public record ModelConfig(
        @JsonProperty("RandomSeed") int randomSeed,
        @JsonProperty("FnnHiddenSizeDeviation") int fnnHiddenSizeDeviation,
        @JsonProperty("TransmissionDelayMin") int transmissionDelayMin,
        @JsonProperty("TransmissionDelayMax") int transmissionDelayMax,
        @JsonProperty("MemoryBlockCapacity") int memoryBlockCapacity,
        @JsonProperty("MemoryAppendageFrequency") int memoryAppendageFrequency,
        @JsonProperty("MemoryBlockFnn") @Nonnull List<Integer> memoryBlockFnn,
        @JsonProperty("Levels") @Nonnull List<Level> levels,
        @JsonProperty("PrimitiveNeuronProcessingFnn") @Nonnull List<Integer> primitiveNeuronProcessingFnn,
        @JsonProperty("PrimitiveNeuronMemorySize") int primitiveNeuronMemorySize
    ) {
        private ModelConfig(@Nonnull Map<String, Object> map) {
            this(
                getNestedField(map, "random_seed"),
                getNestedField(map, "fnn_hidden_size_deviation"),
                getNestedField(map, "transmission_delay", "min"),
                getNestedField(map, "transmission_delay", "max"),
                getNestedField(map, "memory", "block_capacity"),
                getNestedField(map, "memory", "appendage_frequency"),
                getNestedField(map, "memory", "block_fnn"),
                getNestedFieldAsList(Level::new, map, "levels"),
                getNestedField(map, "primitive_neuron", "processing_fnn"),
                getNestedField(map, "primitive_neuron", "memory_size")
            );
        }

        public record Level(
            @JsonProperty("Graph") @Nonnull String graph,
            @JsonProperty("ChildLatentDim") int childLatentDim,
            @JsonProperty("BufferSize") int bufferSize,
            @JsonProperty("MemorySize") int memorySize,
            @JsonProperty("CompositionFnn") @Nonnull List<Integer> compositionFnn,
            @JsonProperty("DecompositionFnn") @Nonnull List<Integer> decompositionFnn
        ) {
            private Level(@Nonnull Map<String, Object> object) {
                this(
                    getNestedField(object, "graph"),
                    getNestedField(object, "child_latent_dim"),
                    getNestedField(object, "buffer_size"),
                    getNestedField(object, "memory_size"),
                    getNestedField(object, "composition_fnn"),
                    getNestedField(object, "decomposition_fnn")
                );
            }
        }
    }

    public record OptimizationConfig(
        @JsonProperty("TrainingFrequency") int trainingFrequency,
        @JsonProperty("TrainingIterations") int trainingIterations,
        @JsonProperty("LearningRate") double learningRate,
        @JsonProperty("BatchSize") int batchSize,
        @JsonProperty("ReplayMemory") int replayMemory
    ) {
        private OptimizationConfig(@Nonnull Map<String, Object> map) {
            this(
                getNestedField(map, "training", "frequency"),
                getNestedField(map, "training", "iterations"),
                getNestedField(map, "learning_rate"),
                getNestedField(map, "batch_size"),
                getNestedField(map, "replay_memory")
            );
        }
    }

    @SuppressWarnings("unchecked")
    private static <R> @Nonnull List<R> getNestedFieldAsList(
        @Nonnull Creator<R> creator,
        Map<String, Object> map,
        @Nonnull String... path
    ) {
        List<?> list = getNestedField(map, path);
        return list.stream()
            .map(o -> (Map<String, Object>) o)
            .map(creator::create)
            .toList();
    }

    @FunctionalInterface
    private interface Creator<T> {
        T create(@Nonnull Map<String, Object> map);
    }

    @SuppressWarnings("unchecked")
    private static <R> R getNestedField(Map<String, Object> map, @Nonnull String... path) {
        if (map == null) {
            throw new NullPointerException("Map cannot be null");
        }

        Object current = map;
        for (String field : path) {
            if (current instanceof Map<?, ?> m) {
                if (!m.containsKey(field)) {
                    throw new NoSuchElementException("Missing field in path " + Arrays.toString(path) + " in config");
                }
                current = m.get(field);
            } else {
                throw new NoSuchElementException("Incorrect field in path " + Arrays.toString(path) + " in config");
            }
        }
        return (R) current;
    }
}
