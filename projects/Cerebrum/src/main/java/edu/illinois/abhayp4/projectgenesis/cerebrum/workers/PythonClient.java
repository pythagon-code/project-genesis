package edu.illinois.abhayp4.projectgenesis.cerebrum.workers;

import jakarta.annotation.Nonnull;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;

final class PythonClient extends PythonExecutor implements Closeable {
    private int port;
    private final Socket socket;
    private final BufferedReader serverIn;
    private final PrintWriter serverOut;

    public PythonClient() {
        try {
            ServerSocket serverSocket = new ServerSocket(0);
            int port = serverSocket.getLocalPort();

            PythonClient client = new PythonClient(port);
            pb.inheritIO();
            process = pb.start();

            socket = serverSocket.accept();
            serverSocket.close();
            InputStreamReader isr = new InputStreamReader(socket.getInputStream());

            serverIn = new BufferedReader(isr);
            serverOut = new PrintWriter(socket.getOutputStream(), true);

            return new PythonClient(0);

        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    public @Nonnull ServerSocket createAndGetSer
        () {
        try {
            return new ServerSocket(0);
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    public PythonClient(int port) {
        super(Integer.toString(port));
    }
}
