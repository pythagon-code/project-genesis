package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers;

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain.SimulatorConfig;
import jakarta.annotation.Nonnull;

public final class ModelWorker extends PythonClient {
    ModelWorker() {}

    public synchronized void initializeClient(@Nonnull SimulatorConfig config) {
        boolean valid = sendAndReceiveObject(
                new DataObjects.ModelInput("InitializeClient", config), boolean.class);

        if (!valid) {
            throw new RuntimeException("Model initialization failed");
        }
    }

    @Override
    public void close() {
        sendObject(new DataObjects.ModelInput("Shutdown", null));
        super.close();
    }

    public synchronized int createModel(@Nonnull String modelType) {
        return sendAndReceiveObject(new DataObjects.ModelInput("CreateModel", modelType), int.class);
    }

    public synchronized @Nonnull DataObjects.ModelTransmissionOutput invokeModel(int modelId) {
        return sendAndReceiveObject(new DataObjects.ModelInput(
            "InvokeModel", modelId), DataObjects.ModelTransmissionOutput.class);
    }

    public static void nothing() {}
}