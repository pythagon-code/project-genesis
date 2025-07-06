package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain;

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

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.application.CerebrumAppContext;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.neurons.RelayNeuron;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.neurons.RelayNeuronFactory;
import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.neurons.ResponseNeuron;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers.ModelWorkerPool;
import jakarta.annotation.Nonnull;

public final class BrainSimulator {
    public final SimulatorConfig config;
    public final ModelWorkerPool modelWorkerPool = null;
    public final RelayNeuronFactory neuronFactory = null;
    private final ObjectWriter writer;
    public final MessageHeartbeat heartbeat;
    private final List<RelayNeuron> relayNeuronList;

    public BrainSimulator(SimulatorSettings settings) {
        ObjectMapper objectMapper = new ObjectMapper();
        DefaultPrettyPrinter prettyPrinter = new DefaultPrettyPrinter();
        DefaultIndenter indenter = new DefaultIndenter("\t", DefaultIndenter.SYS_LF);
        prettyPrinter.indentObjectsWith(indenter);
        prettyPrinter.indentArraysWith(indenter);
        writer = objectMapper.writer(prettyPrinter);

        System.out.println(settings);

        config = new SimulatorConfig(settings);

        System.out.println(config);

        String timestamp = getTimestamp();

        heartbeat = new MessageHeartbeat();
        relayNeuronList = new ArrayList<>();
    }

    public int addNeuron(RelayNeuron neuron) {
        relayNeuronList.add(neuron);
        return relayNeuronList.size();
    }

    public int getNeuronCount() {
        return relayNeuronList.size();
    }

    public void start(CerebrumAppContext feedback) {

    }

    private void saveCheckpoint(ResponseNeuron responseNeuron) {
        String fileName = "hello" + getTimestamp() + ".json";
        String checkpointsFilePath = Paths.get("hello", fileName).toString();

        try (PrintWriter pw = new PrintWriter(checkpointsFilePath)) {
            pw.println(writer.writeValueAsString(null));
        }
        catch (IOException e) {
//            logger.log(Level.SEVERE, "Could not save checkpoint", e);
            throw new IOError(e);
        }
    }

    private @Nonnull String getTimestamp() {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss_z");
        return ZonedDateTime.now().format(formatter);
    }
}