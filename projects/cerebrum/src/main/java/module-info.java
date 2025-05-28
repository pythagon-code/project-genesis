module edu.illinois.abhayp4.projectgenesis.cerebrum {
    requires com.fasterxml.jackson.databind;
    requires jakarta.annotation;
    requires java.logging;
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.graphics;
    requires org.yaml.snakeyaml;

    exports edu.illinois.abhayp4.projectgenesis.cerebrum.api;
    opens edu.illinois.abhayp4.projectgenesis.cerebrum.application to javafx.fxml, javafx.graphics;
}