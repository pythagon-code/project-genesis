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
    apply(plugin = "org.beryx.jlink")

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
        implementation(platform("org.apache.logging.log4j:log4j-bom:2.12.4"))
        implementation("org.apache.logging.log4j:log4j-api")
        implementation("org.apache.logging.log4j:log4j-core")
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

    tasks.register<Copy>("copyDeps") {
        from(configurations.runtimeClasspath)
        into(layout.buildDirectory.dir("deps/"))
    }

    tasks.register<JavaExec>("runModularJar") {
        dependsOn(tasks.build, tasks.getByName("copyDeps"))

        javaLauncher.set(
            javaToolchains.launcherFor {
                languageVersion.set(JavaLanguageVersion.of(23))
            }
        )

        mainModule.set(application.mainModule.get())
        mainClass.set(application.mainClass.get())

        classpath = files()

        val libs = layout.buildDirectory.dir("libs/").get().asFile.absolutePath
        val deps = layout.buildDirectory.dir("deps/").get().asFile.absolutePath

        val args = arrayListOf("--module-path", "$libs${File.pathSeparator}$deps")

        if (project.properties["mode"] == "debug")
            args.add("-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=*:7000")

        jvmArgs = args
    }

    jlink {
        options.set(listOf("--strip-debug", "--no-header-files", "--no-man-pages"))

        launcher {
            name = "app_launcher"
        }

        mergedModule {
            requires("java.logging")
            requires("java.management")
            requires("java.xml")
            requires("org.apache.logging.log4j")
        }
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