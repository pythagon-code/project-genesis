package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.neurons;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.brain.BrainSimulator;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels.PriorityMessageChannel;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels.SourceMessageChannel;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels.TargetMessageChannel;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels.TransmissionMessage;
import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers.ModelWorker;
import jakarta.annotation.Nonnull;

@JsonTypeInfo(
    use = JsonTypeInfo.Id.MINIMAL_CLASS,
    include = JsonTypeInfo.As.PROPERTY
)
@JsonSubTypes({
    @JsonSubTypes.Type(value = PrimitiveNeuron.class),
    @JsonSubTypes.Type(value = MetaNeuron.class),
    @JsonSubTypes.Type(value = ResponseNeuron.class)
})
public sealed abstract class RelayNeuron implements Runnable, Closeable permits MetaNeuron {
    private final SourceMessageChannel source;
    private final List<TargetMessageChannel> targets;
    private BrainSimulator brain = null;
    private final Thread thread;
    protected final ModelWorker modelWorker;
    private int neuronId = -1;
    private volatile boolean awoken = false, done = false;

    public RelayNeuron() {
        source = new PriorityMessageChannel();
        targets = new ArrayList<>();

        thread = new Thread(this, "RelayNeuron-NeuronThread");

        modelWorker = null;
    }

    public void attachBrain(@Nonnull BrainSimulator brain) {
        this.brain = brain;
        neuronId = brain.addNeuron(this);
        brain.heartbeat.registerHeartbeat();
    }

    private void addNeighbor(@Nonnull RelayNeuron neighbor) {
        if (
            targets.contains((TargetMessageChannel) neighbor.source)
            || neighbor.targets.contains((TargetMessageChannel) source)
        ) {
            throw new IllegalArgumentException();
        }

        targets.add((TargetMessageChannel) neighbor.source);
        neighbor.targets.add((TargetMessageChannel) source);
    }

    public void start() {
        thread.start();
    }

    public void awake() {
        awoken = true;
    }

    protected void sendMessage(int channelIdx, String message, double[] latentVector, long targetStep) {
        TargetMessageChannel target = targets.get(channelIdx);
        target.addMessage(new TransmissionMessage(message, latentVector, neuronId, targetStep));
    }

    @Override
    public void run() {
        do {
            brain.heartbeat.awaitProcessMessagePhase();

            while (source.hasAvailableMessage(brain.heartbeat.getStep())) {
                TransmissionMessage message = source.removeMessage();
                awoken = false;
            }

            if (awoken) {
                onAwake();
                awoken = false;
            }

            brain.heartbeat.awaitSendMessagePhase();

            while (source.hasAvailableMessage(brain.heartbeat.getStep())) {
                onMessageReceived(source.removeMessage());
            }
        }
        while (!done);
    }

    @Override
    public void close() {
        done = true;
        try {
            thread.join();
        }
        catch (InterruptedException e) {
            throw new IllegalThreadStateException(e.getMessage());
        }
    }

    protected int getTargetChannelCount() {
        return targets.size();
    }

    protected abstract void onMessageReceived(@Nonnull TransmissionMessage message);

    protected abstract void onAwake();
}
