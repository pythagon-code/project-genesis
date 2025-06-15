package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers;

import jakarta.annotation.Nonnull;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

sealed abstract class PythonExecutor implements Closeable permits PythonClient {
    private static final String pythonExec, workerScript;
    private static final StringBuilder requirementsText = new StringBuilder();

    static {
        try {
            Path tempDir = Files.createTempDirectory("python");
            new File(tempDir.toString()).deleteOnExit();

            String[] resources = {"worker.py", "requirements.txt"};

            for (String resource : resources) {
                try (
                    PrintWriter writer = new PrintWriter(Paths.get(tempDir.toString(), resource).toString());
                    BufferedReader reader = new BufferedReader(new InputStreamReader(
                        Objects.requireNonNull(PythonExecutor.class.getResourceAsStream("/" + resource))));
                ) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        writer.println(line);
                        if (resource.equals("requirements.txt")) {
                            requirementsText.append(line).append("\n");
                        }
                    }
                }
            }

            workerScript = Paths.get(tempDir.toString(), "worker.py").toString();
            String requirements = Paths.get(tempDir.toString(), "requirements.txt").toString();

            if (!Files.isDirectory(Paths.get("gen", ".venv"))) {
                runSafeCommandAndWait("python", "-m", "venv", "gen/.venv");
            }

            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                pythonExec = Paths.get("gen", ".venv", "Scripts", "python.exe").toString();
            } else {
                pythonExec = Paths.get("gen", ".venv", "bin", "python").toString();
            }

            Path initTomlPath = Paths.get("gen", ".venv", "init.toml");

            StringBuilder initTomlText = new StringBuilder();
            if (Files.exists(initTomlPath)) {
                try (BufferedReader reader = new BufferedReader(new FileReader(initTomlPath.toString()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        initTomlText.append(line).append("\n");
                    }
                }
            }

            String wrappedRequirements = '"' + requirementsText.toString().replace("\n", "\\n") + '"';
            if (initTomlText.isEmpty() || !initTomlText.toString().contains(wrappedRequirements)) {
                runSafeCommandAndWait(pythonExec, "-m", "pip", "install", "--upgrade", "pip");
                runSafeCommandAndWait(pythonExec, "-m", "pip", "install", "-r", requirements);
                try (PrintWriter pw = new PrintWriter("gen/.venv/init.toml")) {
                    pw.println("# Created after all Python requirements have been installed");
                    pw.println("[init]");
                    pw.println("time = \"" + ZonedDateTime.now() + "\"");
                    pw.println("requirements = " + wrappedRequirements);
                }
            }
        System.out.println("Python requirements installed");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private Process process = null;

    protected final void startProcess(@Nonnull String... args) {
        if (process != null) {
          throw new RuntimeException("Process already started");
        }

        List<String> scriptAndArgs = new ArrayList<>();
        scriptAndArgs.add(pythonExec);
        scriptAndArgs.add(workerScript);
        scriptAndArgs.addAll(Arrays.asList(args));
        process = runSafeCommand(scriptAndArgs.toArray(new String[0]));
    }

    @Override
    public void close() {
        if (process == null) {
            throw new RuntimeException("Process never started");
        }

        try {
            process.waitFor();
        }
        catch (InterruptedException e) {
            throw new RuntimeException("Cannot interrupt during Python process end");
        }
    }

    private static void runSafeCommandAndWait(@Nonnull String... command) {
        try {
            Process process = runSafeCommand(command);
            process.waitFor();
            if (process.exitValue() != 0) {
                throw new RuntimeException("Command " + Arrays.asList(command) + " failed");
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Cannot interrupt during Python process startProcess");
        }
    }

    private static @Nonnull Process runSafeCommand(@Nonnull String... command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command).inheritIO();
            return pb.start();
        } catch (IOException e) {
            System.err.println(e.getMessage());
            System.exit(1);
            return null;
        }
    }
}
