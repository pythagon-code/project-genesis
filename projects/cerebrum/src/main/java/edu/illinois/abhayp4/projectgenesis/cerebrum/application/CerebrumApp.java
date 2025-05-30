package edu.illinois.abhayp4.projectgenesis.cerebrum.application;

import edu.illinois.abhayp4.projectgenesis.cerebrum.brain.BrainSimulator;
import edu.illinois.abhayp4.projectgenesis.cerebrum.brain.SimulatorSettings;
import edu.illinois.abhayp4.projectgenesis.cerebrum.workers.ModelWorker;
import javafx.application.Application;
import javafx.stage.Stage;

import java.io.*;
import java.util.Properties;

public class CerebrumApp extends Application implements Closeable {
    private final CerebrumAppContext context = new CerebrumAppContext();
    private boolean done = false;

    @Override
    public void start(Stage primaryStage) throws IOException {
        new ModelWorker();

        Properties properties = new Properties();
        SimulatorSettings settings;

        try (InputStream stream = getClass().getResourceAsStream("/default.properties")) {
            if (stream == null)
                throw new IOException();
            properties.load(stream);
            settings = SimulatorSettings.loadFromConfig(properties.getProperty("default.config"));
        } catch (IOException e) {
            throw new IOError(e);
        }

        new Thread(() -> new BrainSimulator(settings).start(context), "BrainSimulator-Thread");

        primaryStage.show();
    }

    @Override
    public void close() {
        done = true;
    }
}
