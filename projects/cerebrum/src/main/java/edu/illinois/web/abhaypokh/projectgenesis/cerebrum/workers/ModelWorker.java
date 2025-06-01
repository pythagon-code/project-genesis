package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers;

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain.SimulatorConfig;

public final class ModelWorker extends PythonClient {
    public synchronized boolean initializeClient(SimulatorConfig config) {
        return sendAndReceiveObject(new DataObjects.ModelInput("InitializeClient", config), boolean.class);
    }

    @Override
    public void close() {
        sendObject(new DataObjects.ModelInput("Shutdown", null));
        super.close();
    }

    public synchronized int createModel(String modelType) {
        return sendAndReceiveObject(new DataObjects.ModelInput("CreateModel", modelType), int.class);
    }

    public synchronized DataObjects.ModelTransmissionOutput invokeModel(int modelId) {
        return sendAndReceiveObject(new DataObjects.ModelInput(
            "InvokeModel", modelId), DataObjects.ModelTransmissionOutput.class);

    }
}