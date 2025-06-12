package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels;

import com.fasterxml.jackson.annotation.JsonIgnoreType;

@JsonIgnoreType
public sealed interface SourceMessageChannel permits BasicMessageChannel, PriorityMessageChannel {
    TransmissionMessage removeMessage();
    boolean hasAvailableMessage(long currentStep);
}