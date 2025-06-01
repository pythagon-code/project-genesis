package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.Nonnull;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;

sealed class PythonClient extends PythonExecutor implements Closeable permits ModelWorker {
    private final Socket socket;
    private final BufferedReader serverIn;
    private final PrintWriter serverOut;
    private final ObjectMapper mapper;

    public PythonClient() {
        try {
            ServerSocket serverSocket = new ServerSocket(0);
            int port = serverSocket.getLocalPort();

            startProcess(Integer.toString(port));

            socket = serverSocket.accept();
            serverSocket.close();
            InputStreamReader isr = new InputStreamReader(socket.getInputStream());

            serverIn = new BufferedReader(isr);
            serverOut = new PrintWriter(socket.getOutputStream(), true);

            mapper = new ObjectMapper();
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    public <R> R sendAndReceiveObject(@Nonnull Object data, @Nonnull Class<R> clazz) {
        sendObject(data);
        return receiveObject(clazz);
    }

    protected final void sendObject(@Nonnull Object data) {
        try {
            mapper.writeValueAsString(data);
        } catch (JsonProcessingException e) {
            throw new IOError(e);
        }
        send(data.toString());
    }

    private @Nonnull <R> R receiveObject(@Nonnull Class<R> clazz) {
        try {
            return mapper.readValue(receive(), clazz);
        }
        catch (JsonProcessingException e) {
            throw new IOError(e);
        }
    }

    private void send(@Nonnull String message) {
        serverOut.println(message);
    }

    private synchronized @Nonnull String receive() {
        try {
            String line = serverIn.readLine();
            if (line == null) {
                throw new RuntimeException();
            }
            return line;
        } catch (IOException e) {
            throw new IOError(e);
        }
    }

    @Override
    public void close() {
        try {
            serverIn.close();
        }
        catch (IOException e) {
            System.err.println(e.getMessage());
        }
        finally {
            serverOut.close();
            try {
                socket.close();
            }
            catch (IOException e2) {
                System.err.println(e2.getMessage());
            }
        }

        super.close();
    }
}
