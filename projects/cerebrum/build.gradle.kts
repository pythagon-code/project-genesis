plugins {
    application
    id("org.openjfx.javafxplugin") version "0.1.0"
    id("org.beryx.jlink") version "3.1.1"
}

group = "edu.illinois.abhayp4.projectgenesis.cerebrum"
version = "1.0-SNAPSHOT"

dependencies {
    implementation("org.yaml:snakeyaml:2.3")
    implementation(platform("com.fasterxml.jackson:jackson-bom:2.18.0"))
    implementation("com.fasterxml.jackson.core:jackson-core")
    implementation("com.fasterxml.jackson.core:jackson-annotations")
    implementation("com.fasterxml.jackson.core:jackson-databind")
}

sourceSets {
    main {
        resources {
            srcDirs("src/main/resources/", "src/main/python/")
        }
    }
}

application {
    mainModule.set("edu.illinois.abhayp4.projectgenesis.cerebrum")
    mainClass.set("edu.illinois.abhayp4.projectgenesis.cerebrum.api.CerebrumApi")
}

javafx {
    version = "24.0.1"
    modules("javafx.controls", "javafx.fxml", "javafx.graphics")
}

jlink {
    options.set(listOf("--no-header-files", "--no-man-pages"))

    launcher {
        name = "project-genesis-cerebrum"
    }

    addExtraDependencies("javafx")
}