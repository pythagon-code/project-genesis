package edu.illinois.abhayp4.projectgenesis.cerebrum.brain;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.annotation.Nonnull;

import java.util.Arrays;
import java.util.Map;
import java.util.NoSuchElementException;

public record SimulatorConfig(
    @JsonProperty("SystemConfig") SystemConfig systemConfig,
    @JsonProperty("ModelArchitectureConfig") @Nonnull ModelArchitectureConfig modelArchitectureConfig,
    @JsonProperty("TransformersConfig") @Nonnull TransformersConfig transformersConfig,
    @JsonProperty("NeuronTopologyConfig") @Nonnull NeuronTopologyConfig neuronTopologyConfig,
    @JsonProperty("BaseNeuronConfig") @Nonnull BaseNeuronConfig baseNeuronConfig,
    @JsonProperty("GraphStructuresConfig") @Nonnull GraphStructuresConfig graphStructuresConfig,
    @JsonProperty("OptimizationConfig") @Nonnull OptimizationConfig optimizationConfig
) {
    public SimulatorConfig(@Nonnull SimulatorSettings settings) {
        this(
            settings.systemObject(),
            settings.modelArchitectureObject(),
            settings.transformersObject(),
            settings.neuronTopologyObject(),
            settings.baseNeuronObject(),
            settings.graphStructuresObject(),
            settings.optimizationObject()
        );
    }

    @SuppressWarnings("unchecked")
    private SimulatorConfig(
        Map<String, Object> systemObject,
        Map<String, Object> modelArchitectureObject,
        Map<String, Object> transformersObject,
        Map<String, Object> neuronTopologyObject,
        Map<String, Object> baseNeuronObject,
        Map<String, Object> graphStructuresObject,
        Map<String, Object> optimizationObject
    ) {
        this(
            new SystemConfig((Map<String, Object>) systemObject.get("system")),
            new ModelArchitectureConfig((Map<String, Object>) modelArchitectureObject.get("modelArchitecture")),
            new TransformersConfig((Map<String, Object>) transformersObject.get("transformers")),
            new NeuronTopologyConfig((Map<String, Object>) neuronTopologyObject.get("neuronTopology")),
            new BaseNeuronConfig((Map<String, Object>) baseNeuronObject.get("baseNeuron")),
            new GraphStructuresConfig((Map<String, Object>) graphStructuresObject.get("graphStructures")),
            new OptimizationConfig((Map<String, Object>) optimizationObject.get("optimization"))
        );
    }

    public record SystemConfig(
        @JsonProperty("PythonExecutable") @Nonnull String pythonExecutable,
        @JsonProperty("NPythonWorkers") int nPythonWorkers,
        @JsonProperty("UseCuda") boolean useCuda,
        @JsonProperty("CudaDevice") int cudaDevice,
        @JsonProperty("TrainingAllowed") boolean trainingAllowed,
        @JsonProperty("PreloadModelEnabled") boolean preloadModelEnabled,
        @JsonProperty("PreloadModelFrom") @Nonnull String preloadModelFrom,
        @JsonProperty("ErrorOnInconsistentModel") boolean errorOnInconsistentModel,
        @JsonProperty("ErrorOnDifferentOptimization") boolean errorOnDifferentOptimization,
        @JsonProperty("SaveCheckpointsEnabled") boolean saveCheckpointsEnabled,
        @JsonProperty("SaveCheckpointsTo") @Nonnull String saveCheckpointsTo,
        @JsonProperty("SaveCheckpointsFileNamePrefix") @Nonnull String saveCheckpointsFileNamePrefix,
        @JsonProperty("SaveCheckpointsFrequency") int saveCheckpointsFrequency,
        @JsonProperty("LogTo") @Nonnull String logTo,
        @JsonProperty("LogFileNamePrefix") @Nonnull String logFileNamePrefix,
        @JsonProperty("NRotatingLogs") int nRotatingLogs,
        @JsonProperty("MaxSizePerLog") int maxSizePerLog,
        @JsonProperty("LogVerbosity") @Nonnull LogVerbosity logVerbosity
    ) {
        private SystemConfig (Map<String, Object> object) {
            this(
                getNestedField(object, "python_executable"),
                getNestedField(object, "n_python_workers"),
                getNestedField(object, "use_cuda"),
                getNestedField(object, "cuda_device"),
                getNestedField(object, "training_allowed"),
                getNestedField(object, "preload_model", "enabled"),
                getNestedField(object, "preload_model", "from"),
                getNestedField(object, "preload_model", "error_on_inconsistent_model"),
                getNestedField(object, "preload_model", "error_on_different_optimization"),
                getNestedField(object, "save_checkpoints", "enabled"),
                getNestedField(object, "save_checkpoints", "to"),
                getNestedFieldOrDefault(object, "", "save_checkpoints", "file_name_prefix"),
                getNestedField(object, "save_checkpoints", "frequency"),
                getNestedField(object, "log", "to"),
                getNestedFieldOrDefault(object, "", "log", "file_name_prefix"),
                getNestedField(object, "log", "n_rotating_logs"),
                getNestedField(object, "log", "max_size_per_log"),
                LogVerbosity.valueOf(((String) getNestedField(object, "log", "verbosity")).toUpperCase())
            );
        }
    }

    public enum LogVerbosity {
        LOW,
        MEDIUM,
        HIGH
    };

    public record ModelArchitectureConfig(
        @JsonProperty("PythonExecutable") String pythonExecutable,
        @JsonProperty("PythonExecutable") String pythonExecutable2
    ) {
        private ModelArchitectureConfig(Map<String, Object> object) {
            this(
                getNestedField(object, "main_config", "python_executable"),
                getNestedField(object, "main_config", "python_executable2")
            );
        }
    }

    public record TransformersConfig(
        @JsonProperty("PythonExecutable") String pythonExecutable,
        @JsonProperty("PythonExecutable") String pythonExecutable2
    ) {
        TransformersConfig(Map<String, Object> object) {
            this(
                getNestedField(object, "main_config", "python_executable"),
                getNestedField(object, "main_config", "python_executable2")
            );
        }
    }

    public record NeuronTopologyConfig(
        @JsonProperty("PythonExecutable") String pythonExecutable,
        @JsonProperty("PythonExecutable") String pythonExecutable2
    ) {
        private NeuronTopologyConfig(Map<String, Object> object) {
            this(
                getNestedField(object, "main_config", "python_executable"),
                getNestedField(object, "main_config", "python_executable2")
            );
        }
    }

    public record BaseNeuronConfig(
        @JsonProperty("PythonExecutable") String pythonExecutable,
        @JsonProperty("PythonExecutable") String pythonExecutable2
    ) {
        private BaseNeuronConfig(Map<String, Object> object) {
            this(
                getNestedField(object, "main_config", "python_executable"),
                getNestedField(object, "main_config", "python_executable2")
            );
        }
    }

    public record GraphStructuresConfig(
        @JsonProperty("PythonExecutable") @Nonnull String pythonExecutable,
        @JsonProperty("PythonExecutable") @Nonnull String pythonExecutable2
    ) {
        private GraphStructuresConfig(Map<String, Object> object) {
            this(
                getNestedField(object, "main_config", "python_executable"),
                getNestedField(object, "main_config", "python_executable2")
            );
        }
    }

    public record OptimizationConfig(
        @JsonProperty("PythonExecutable") String pythonExecutable,
        @JsonProperty("PythonExecutable") String pythonExecutable2
    ) {
        private OptimizationConfig(Map<String, Object> object) {
            this(
                getNestedField(object, "main_config", "python_executable"),
                getNestedField(object, "main_config", "python_executable2")
            );
        }
    }

    @SuppressWarnings("unchecked")
    private static <R> R getNestedFieldOrDefault(Map<String, Object> object, R defaultValue, String... path) {
        Object result = getNestedField(object, path);
        if (result == null) {
            return defaultValue;
        }
        else {
            return (R) result;
        }
    }

    @SuppressWarnings("unchecked")
    private static <R> R getNestedField(Map<String, Object> object, String... path) {
        Object current = object;
        for (String field : path) {
            if (!(current instanceof Map)) {
                throw new NoSuchElementException("Missing field in path " + Arrays.toString(path) + " in config");
            }
            current = ((Map<String, Object>) current).get(field);
        }
        return (R) current;
    }
}
