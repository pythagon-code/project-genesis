package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels;

import com.fasterxml.jackson.annotation.JsonIgnoreType;

@JsonIgnoreType
public sealed interface TargetMessageChannel permits BasicMessageChannel, MessageChannel {
    void addMessage(TransmissionMessage message);
}
