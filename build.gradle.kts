plugins {
    application
    id("org.openjfx.javafxplugin") version "0.1.0"
    id("org.beryx.jlink") version "3.1.1"
}

group = "edu.illinois.web.abhaypokh.projectgenesis"
version = "1.0-SNAPSHOT"

allprojects {
    apply(plugin = "java")
    apply(plugin = "application")

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(23))
        }
        modularity.inferModulePath = true
    }

    repositories {
        mavenCentral()
    }

    dependencies {
        testImplementation("org.junit.jupiter:junit-jupiter:5.7.1")
        testRuntimeOnly("org.junit.platform:junit-platform-launcher")
        implementation("jakarta.annotation:jakarta.annotation-api:2.1.1")
        implementation("org.slf4j:slf4j-api:2.0.12")
        implementation("ch.qos.logback:logback-classic:1.4.14")
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
        maxHeapSize = "1G"

        testLogging {
            events("passed", "failed", "skipped")
            showStandardStreams = true
        }
    }

    tasks.withType<ProcessResources> {
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    }

    tasks.withType<Jar> {
        doLast {
            manifest {
                attributes["Main-Class"] = application.mainClass.get()
            }
        }
    }

    tasks.register<Copy>("copyDeps") {
        from(configurations.runtimeClasspath)
        into(layout.buildDirectory.dir("deps/"))
    }

    tasks.register<JavaExec>("runModularJar") {
        dependsOn(tasks.getByName("build"), tasks.getByName("copyDeps"))

        javaLauncher.set(
            javaToolchains.launcherFor {
                languageVersion.set(JavaLanguageVersion.of(23))
            }
        )

        mainModule.set(application.mainModule.get())
        mainClass.set(application.mainClass.get())

        val libs = layout.buildDirectory.dir("libs/").get().asFile
        val deps = layout.buildDirectory.dir("deps/").get().asFile

        classpath = files(libs, deps)

        jvmArgs = listOf(
            "--module-path", "$libs${File.pathSeparator}$deps",
            "-m", "${application.mainModule.get()}/${application.mainClass.get()}"
        )
    }
}

dependencies {
    implementation(project(":cerebrum"))
    implementation(project(":new-eden"))
}

application {
    mainModule.set("edu.illinois.web.abhaypokh.projectgenesis")
    mainClass.set("edu.illinois.web.abhaypokh.projectgenesis.application.Main")
}

jlink {
    options.set(listOf("--no-header-files", "--no-man-pages"))

    launcher {
        name = "project_genesis_launcher"
    }

    addExtraDependencies("javafx")
}