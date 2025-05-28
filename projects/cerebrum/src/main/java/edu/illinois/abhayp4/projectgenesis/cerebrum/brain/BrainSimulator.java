package edu.illinois.abhayp4.projectgenesis.cerebrum.brain;

import java.io.File;
import java.io.IOError;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Paths;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import edu.illinois.abhayp4.projectgenesis.cerebrum.application.CerebrumAppContext;
import edu.illinois.abhayp4.projectgenesis.cerebrum.neurons.RelayNeuron;
import edu.illinois.abhayp4.projectgenesis.cerebrum.neurons.RelayNeuronFactory;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import edu.illinois.abhayp4.projectgenesis.cerebrum.neurons.ResponseNeuron;
import edu.illinois.abhayp4.projectgenesis.cerebrum.workers.ModelWorkerPool;

public final class BrainSimulator {
    public final SimulatorConfig config;
    public final ModelWorkerPool modelWorkerPool = null;
    public final RelayNeuronFactory neuronFactory = null;
    public final Logger logger;
    public final String checkpointsFolderPath;
    private final ObjectWriter writer;
    public final MessageHeartbeat heartbeat;
    private final List<RelayNeuron> relayNeuronList;

    public static final Level LOW = new Level("LOW", Level.INFO.intValue() + 1) { };
    public static final Level MEDIUM = new Level("MEDIUM", LOW.intValue() + 1) { };
    public static final Level HIGH = new Level("HIGH", MEDIUM.intValue() + 1) { };

    public BrainSimulator(SimulatorSettings settings) {
        config = new SimulatorConfig(settings);

        ObjectMapper objectMapper = new ObjectMapper();
        DefaultPrettyPrinter prettyPrinter = new DefaultPrettyPrinter();
        DefaultIndenter indenter = new DefaultIndenter("\t", DefaultIndenter.SYS_LF);
        prettyPrinter.indentObjectsWith(indenter);
        prettyPrinter.indentArraysWith(indenter);
        writer = objectMapper.writer(prettyPrinter);
        
        String timestamp = getTimestamp();
        
        String logFolderPath = Paths.get(
            config.systemConfig().logTo().replace("%", "%%"), timestamp).toString();
        if (!new File(logFolderPath).mkdirs()) {
            throw new NoSuchElementException("Could not create log folder " + logFolderPath);
        }

        String logFileName = config.systemConfig().logFileNamePrefix().replace("%", "%%") + "%g.log";
        String logFilePath = Paths.get(logFolderPath, logFileName).toString();

        try {
            final int BYTES_PER_MB = 1024 * 1024;
            FileHandler fh = new FileHandler(
                logFilePath,
                BYTES_PER_MB * config.systemConfig().maxSizePerLog(), config.systemConfig().nRotatingLogs(),
                true
            );
            fh.setFormatter(new SimpleFormatter());

            logger = Logger.getLogger("brain-simulation");
            logger.addHandler(fh);
        }
        catch (IOException e) {
            throw new IOError(e);
        }

        switch (config.systemConfig().logVerbosity()) {
            case SimulatorConfig.LogVerbosity.LOW:
                logger.setLevel(LOW);
                break;
            case SimulatorConfig.LogVerbosity.MEDIUM:
                logger.setLevel(MEDIUM);
                break;
            case SimulatorConfig.LogVerbosity.HIGH:
                logger.setLevel(HIGH);
        }

        logger.addHandler(new ConsoleHandler());
        logger.setUseParentHandlers(true);
        
        if (config.systemConfig().saveCheckpointsEnabled()) {
            checkpointsFolderPath = Paths.get(config.systemConfig().saveCheckpointsTo(), timestamp).toString();
            File checkpointsFolder = new File(checkpointsFolderPath);
            if (!checkpointsFolder.mkdirs()) {
                throw new NoSuchElementException("Could not create checkpoint folder " + checkpointsFolderPath);
            }
        } else {
            checkpointsFolderPath = null;
        }

        heartbeat = new MessageHeartbeat();
        relayNeuronList = new ArrayList<>();
    }

    public int addNeuron(RelayNeuron neuron) {
        relayNeuronList.add(neuron);
        return relayNeuronList.size();
    }

    public void start(CerebrumAppContext feedback) {

    }

    private void saveCheckpoint(ResponseNeuron responseNeuron) {
        if (!config.systemConfig().saveCheckpointsEnabled()) {
            return;
        }

        String fileName = config.systemConfig().saveCheckpointsFileNamePrefix() + getTimestamp() + ".json";
        String checkpointsFilePath = Paths.get(checkpointsFolderPath, fileName).toString();

        try (PrintWriter pw = new PrintWriter(checkpointsFilePath)) {
            pw.println(writer.writeValueAsString(null));
        }
        catch (IOException e) {
            logger.log(Level.SEVERE, "Could not save checkpoint", e);
            throw new IOError(e);
        }
    }

    private String getTimestamp() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss_z");
        return ZonedDateTime.now().format(formatter);
    }
}