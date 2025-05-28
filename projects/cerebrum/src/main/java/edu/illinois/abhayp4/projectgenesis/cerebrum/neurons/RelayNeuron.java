package edu.illinois.abhayp4.projectgenesis.cerebrum.neurons;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import edu.illinois.abhayp4.projectgenesis.cerebrum.brain.BrainSimulator;
import edu.illinois.abhayp4.projectgenesis.cerebrum.channels.MessageChannel;
import edu.illinois.abhayp4.projectgenesis.cerebrum.channels.SourceMessageChannel;
import edu.illinois.abhayp4.projectgenesis.cerebrum.channels.TargetMessageChannel;
import edu.illinois.abhayp4.projectgenesis.cerebrum.channels.TransmissionMessage;
import edu.illinois.abhayp4.projectgenesis.cerebrum.workers.ModelWorker;
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
    private final ModelWorker modelWorker;
    private int neuronId = -1;
    private volatile boolean awaken = false;
    private boolean done = false;

    public RelayNeuron() {
        source = new MessageChannel();
        targets = new ArrayList<>();

        thread = new Thread(this, "RelayNeuron-NeuronThread");

        modelWorker = null;
    }

    public void attachBrain(@Nonnull BrainSimulator brain) {
        this.brain = brain;
        neuronId = brain.addNeuron(this);
        brain.heartbeat.registerHeartbeat();
    }

    public void addTarget(TargetMessageChannel target) {
        if (targets.contains(target)) {
            throw new IllegalArgumentException();
        }

        targets.add(target);
    }

    public void start() {
        thread.start();
    }

    public void awake() {
        awaken = true;
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
                awaken = false;
            }

            if (awaken) {
                onAwaken();
                awaken = false;
            }

            brain.heartbeat.awaitSendMessagePhase();

            while (source.hasAvailableMessage(brain.heartbeat.getStep())) {
                onMessageReceived(source.removeMessage());
            }
        }
        while (!done);
    }

    @Override
    public synchronized void close() {
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

    protected abstract void onAwaken();
}
