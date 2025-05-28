package edu.illinois.abhayp4.projectgenesis.cerebrum.channels;

import java.util.PriorityQueue;
import java.util.Queue;

import com.fasterxml.jackson.annotation.JsonIgnoreType;

@JsonIgnoreType
public final class MessageChannel implements SourceMessageChannel, TargetMessageChannel {
    private final Queue<TransmissionMessage> queue;

    public MessageChannel() {
        queue = new PriorityQueue<>();
    }

    @Override
    public TransmissionMessage removeMessage() {
        return queue.remove();
    }

    @Override
    public boolean hasAvailableMessage(long currentStep) {
        return !queue.isEmpty() && queue.peek().targetStep() == currentStep;
    }

    @Override
    public void addMessage(TransmissionMessage message) {
        queue.add(message);
    }
}