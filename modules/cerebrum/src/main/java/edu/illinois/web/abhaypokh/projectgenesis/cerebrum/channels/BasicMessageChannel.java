package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels;

import jakarta.annotation.Nonnull;

import java.util.LinkedList;
import java.util.List;

public final class BasicMessageChannel implements TargetMessageChannel {
    private final List<TransmissionMessage> list = new LinkedList<>();
    private final int bufferSize;

    public BasicMessageChannel(int bufferSize) {
        this.bufferSize = bufferSize;
    }

    @Override
    public void addMessage(TransmissionMessage message) {
        list.add(message);
    }

    public boolean exceedsBufferSize() {
        return list.size() >= bufferSize;
    }

    public @Nonnull Iterable<TransmissionMessage> getMessages() {
        return list.subList(0, bufferSize);
    }
}