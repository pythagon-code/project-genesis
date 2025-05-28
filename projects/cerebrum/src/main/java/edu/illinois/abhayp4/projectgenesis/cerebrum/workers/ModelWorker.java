package edu.illinois.abhayp4.projectgenesis.cerebrum.workers;

import edu.illinois.abhayp4.projectgenesis.cerebrum.brain.SimulatorConfig;

public final class ModelWorker extends PythonClient {
    public synchronized boolean initialize(SimulatorConfig config) {
        return sendAndReceiveObject(new DataObjects.ModelInput("Initialize", config), boolean.class);
    }

    @Override
    public void close() {
        sendObject(new DataObjects.ModelInput("Shutdown", null));
        super.close();
    }

    public int createModel(String modelType) {
        return sendAndReceiveObject(new DataObjects.ModelInput("CreateModel", modelType), int.class);
    }

    public DataObjects.ModelOutput invokeModel() {
        return null;
    }
}