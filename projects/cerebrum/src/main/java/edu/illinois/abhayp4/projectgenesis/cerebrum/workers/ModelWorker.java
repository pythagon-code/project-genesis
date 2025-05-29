package edu.illinois.abhayp4.projectgenesis.cerebrum.workers;

import edu.illinois.abhayp4.projectgenesis.cerebrum.brain.SimulatorConfig;

public final class ModelWorker extends PythonClient {
    public synchronized boolean initialize(SimulatorConfig config) {
        return sendAndReceiveObject(new DataObjects.ModelInput("InitializeClient", config), boolean.class);
    }

    @Override
    public synchronized void close() {
        sendObject(new DataObjects.ModelInput("Shutdown", null));
        super.close();
    }

    public synchronized int createModel(String modelType) {
        return sendAndReceiveObject(new DataObjects.ModelInput("CreateModel", modelType), int.class);
    }

    public synchronized DataObjects.ModelTransmissionOutput invokeModel(int modelId) {
        return sendAndReceiveObject(new DataObjects.ModelInput(
            "InvokeModel", modelId-), DataObjects.ModelTransmissionOutput.class);

    }
}