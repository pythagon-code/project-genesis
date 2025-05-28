package edu.illinois.abhayp4.projectgenesis.cerebrum.brain;

import java.util.concurrent.Phaser;

public final class MessageHeartbeat {
    private final Phaser processMessageBarrier, sendMessageBarrier;
    private long step = 0;

    private boolean running = false;

    public MessageHeartbeat() {
        processMessageBarrier = new Phaser(1);
        sendMessageBarrier = new Phaser(0);
    }

    public synchronized void registerHeartbeat() {
        if (running) {
            throw new IllegalStateException("Cannot register heartbeat during heartbeat loop");
        }

        processMessageBarrier.register();
        sendMessageBarrier.register();
    }

    public void awaitProcessMessagePhase() {
        processMessageBarrier.arriveAndAwaitAdvance();
    }

    public void awaitSendMessagePhase() {
        sendMessageBarrier.arriveAndAwaitAdvance();
    }

    public long getStep() {
        return step;
    }

    synchronized void step() {
        running = true;
        processMessageBarrier.arrive();

        do {
            Thread.onSpinWait();
        } while (processMessageBarrier.getUnarrivedParties() != 1);

        step++;
    }
}