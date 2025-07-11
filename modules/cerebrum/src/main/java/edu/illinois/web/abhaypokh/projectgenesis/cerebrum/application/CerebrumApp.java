package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.application;

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain.BrainSimulator;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain.SimulatorSettings;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers.ModelWorker;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers.ModelWorkerPool;
import javafx.application.Application;
import javafx.stage.Stage;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.util.Properties;

public class CerebrumApp extends Application implements Closeable {
    private final CerebrumAppContext context = new CerebrumAppContext();
    private boolean done = false;

    @Override
    public void start(Stage primaryStage) throws IOException {
        Logger logger = LogManager.getLogger(getClass());
        for (int i = 0; i < 1; i++) {
            logger.info("Hello World!");
            logger.debug("Hello World!");
            logger.warn("Hello World!");
        }

        ModelWorkerPool.nothing();

        System.out.println(System.getProperty("user.dir"));

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

        BrainSimulator brain = new BrainSimulator(settings);

        new Thread(() -> brain.start(context), "BrainSimulator-Thread");

        primaryStage.show();

        System.out.println("Done");
    }

    @Override
    public void close() {
        done = true;
    }
}
