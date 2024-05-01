import 'dart:convert';
import 'dart:io';

import 'package:change_case/change_case.dart';
import 'package:logging/logging.dart';
import 'package:native_assets_cli/native_assets_cli.dart';
import 'package:path/path.dart' as p;

abstract class Builder {
  Future<void> run({
    required BuildConfig buildConfig,
    required BuildOutput buildOutput,
    required Logger? logger,
  });
}

class CMakeBuilder implements Builder {
  /// Name of the library or executable to build.
  ///
  /// The filename will be decided by [BuildConfig.targetOS] and
  /// [OS.libraryFileName] or [OS.executableFileName].
  ///
  /// File will be placed in [BuildConfig.outputDirectory].
  final String name;

  /// Asset identifier.
  ///
  /// Used to output the [BuildOutput.assets].
  ///
  /// If omitted, no asset will be added to the build output.
  final String? assetName;

  /// Include directories to pass to the compiler.
  ///
  /// Resolved against [BuildConfig.packageRoot].
  ///
  /// Used to output the [BuildOutput.dependencies].
  final List<String> includes;

  final String? preset;

  /// Definitions of preprocessor macros.
  ///
  /// When the value is `null`, the macro is defined without a value.
  final Map<String, String?> defines;

  /// The dart files involved in building this artifact.
  ///
  /// Resolved against [BuildConfig.packageRoot].
  ///
  /// Used to output the [BuildOutput.dependencies].
  final List<String> dartBuildFiles;

  /// The language standard to use.
  ///
  /// When set to `null`, the default behavior of the compiler will be used.
  final String? std;

  final File? userToolchain;
  final Directory sourceDirectory;

  CMakeBuilder.library({
    required this.name,
    required this.sourceDirectory,
    this.assetName,
    this.includes = const [],
    this.preset,
    this.defines = const {},
    this.dartBuildFiles = const [],
    this.std,
    this.userToolchain,
  });

  @override
  Future<void> run({
    required BuildConfig buildConfig,
    required BuildOutput buildOutput,
    required Logger? logger,
  }) async {
    final outDir = buildConfig.outputDirectory;
    final packageRoot = buildConfig.packageRoot;
    await Directory.fromUri(outDir).create(recursive: true);
    final linkMode = _linkMode(buildConfig.linkModePreference);
    final targetOS = buildConfig.targetOS;
    // final libUri = outDir.resolve(buildConfig.targetOS.libraryFileName(name, linkMode));
    // final exeUri = outDir.resolve(buildConfig.targetOS.executableFileName(name));
    // final sources = packageRoot.resolveUri(Uri.file(source));
    // final includes = [
    //   for (final directory in this.includes) packageRoot.resolveUri(Uri.file(directory)),
    // ];
    // final dartBuildFiles = [
    //   for (final source in this.dartBuildFiles) packageRoot.resolve(source),
    // ];

    final errMsg = "OS: $targetOS, Architecture: ${buildConfig.targetArchitecture} not supported";
    defines["CMAKE_INSTALL_PREFIX"] = buildConfig.outputDirectory.resolve("install").toFilePath();
    defines["GGML_BUILD_EXAMPLES"] = "OFF";
    defines["GGML_BUILD_TESTS"] = "OFF";
    switch (targetOS) {
      case OS.iOS:
        assert(userToolchain != null, "userToolchain is null");
        defines["CMAKE_TOOLCHAIN_FILE"] = userToolchain!.path;
      case OS.android:
        final toolchainPtn = "../../../../../build/cmake/android.toolchain.cmake";
        final ndkTc = buildConfig.cCompiler.compiler?.resolve(toolchainPtn);
        if (ndkTc == null || !File.fromUri(ndkTc).existsSync()) {
          throw p.PathException("android.toolchain.cmake not found!");
        }
        final abi = switch (buildConfig.targetArchitecture) {
          Architecture.arm => "armeabi-v7a",
          Architecture.arm64 => "arm64-v8a",
          Architecture.x64 => "x86_64",
          null => buildConfig.dryRun ? "x86_64" : throw UnsupportedError(errMsg),
          _ => throw UnsupportedError(errMsg),
        };
        defines["CMAKE_TOOLCHAIN_FILE"] = ndkTc.toFilePath();
        defines["ANDROID_ABI"] = abi;
        defines["ANDROID_PLATFORM"] = "${buildConfig.targetAndroidNdkApi}";
      case OS.windows:
        defines["CXX_STANDARD "] = "20";
      default:
    }
    if (buildConfig.dryRun) {
      final result = await runProcess(
        executable: "cmake",
        arguments: ["--version"],
        logger: logger,
      );
      if (result.exitCode != 0) throw CMakeBuildException("cmake not installed");
    } else {
      final defs = defines.entries.map((e) => "-D${e.key}=${e.value}");
      // generate
      final resGen = await runProcess(
        executable: "cmake",
        arguments: [
          "-S",
          p.normalize(sourceDirectory.path),
          "-B",
          p.normalize(outDir.toFilePath()),
          ...defs,
        ],
        logger: logger,
        workingDirectory: outDir,
        captureOutput: false,
        throwOnUnexpectedExitCode: true,
      );
      if (resGen.exitCode != 0) throw CMakeBuildException("Generation failed, exit code: ${resGen.exitCode}");

      // build
      final resBuild = await runProcess(
        executable: "cmake",
        arguments: [
          "--build",
          p.normalize(outDir.toFilePath()),
          "--config",
          buildConfig.buildMode.name.toCapitalCase(),
          "--target",
          "install",
        ],
        logger: logger,
        workingDirectory: outDir,
        captureOutput: false,
        throwOnUnexpectedExitCode: true,
      );
      if (resBuild.exitCode != 0) throw CMakeBuildException("Build failed, exit code: ${resBuild.exitCode}");
    }

    final ext = switch (targetOS) {
      OS.windows => "dll",
      OS.linux => "so",
      OS.macOS => "dylib",
      OS.android => "so",
      OS.fuchsia => "so",
      OS.iOS => "framework",
      OS() => throw UnimplementedError(),
    };
    final installDir = Directory.fromUri(outDir.resolve("install/bin"));
    if (!installDir.existsSync()) {
      installDir.createSync(recursive: true);
    }
    final libEntities = installDir.listSync();

    if (assetName != null) {
      final assets = <Asset>[];
      for (var entity in libEntities) {
        final fileName = p.basename(entity.path);
        Uri assetUri;
        // for windows, linux, android, the final lib is a file,
        // i.e., .dll, .so files
        if (entity is File && entity.path.endsWith(ext)) {
          assetUri = entity.uri;
        }
        // for ios, the final lib is .framework, i.e., a directory,
        else if (entity is Directory && targetOS == OS.iOS) {
          assetUri = entity.uri.resolve(name);
        } else {
          continue;
        }
        assets.add(NativeCodeAsset(
          package: buildConfig.packageName,
          name:
              fileName.contains(buildConfig.packageName) ? assetName! : p.basenameWithoutExtension(fileName),
          file: assetUri,
          linkMode: linkMode,
          os: targetOS,
          architecture: buildConfig.dryRun ? null : buildConfig.targetArchitecture,
        ));
      }
      buildOutput.addAssets(assets);
    }
    // if (!buildConfig.dryRun) {
    //   final includeFiles = await Stream.fromIterable(includes)
    //       .asyncExpand(
    //         (include) => Directory(include.toFilePath())
    //             .list(recursive: true)
    //             .where((entry) => entry is File)
    //             .map((file) => file.uri),
    //       )
    //       .toList();

    //   // buildOutput.addDependencies({
    //   //   // Note: We use a Set here to deduplicate the dependencies.
    //   //   // ...sources,
    //   //   ...includeFiles,
    //   //   ...dartBuildFiles,
    //   // });
    // }
  }
}

LinkMode _linkMode(LinkModePreference preference) {
  if (preference == LinkModePreference.dynamic || preference == LinkModePreference.preferDynamic) {
    return DynamicLoadingBundled();
  }
  assert(preference == LinkModePreference.static || preference == LinkModePreference.preferStatic);
  return StaticLinking();
}

/// Runs a [Process].
///
/// If [logger] is provided, stream stdout and stderr to it.
///
/// If [captureOutput], captures stdout and stderr.
Future<RunProcessResult> runProcess({
  required String executable,
  List<String> arguments = const [],
  Uri? workingDirectory,
  Map<String, String>? environment,
  bool includeParentEnvironment = true,
  required Logger? logger,
  bool captureOutput = true,
  int expectedExitCode = 0,
  bool throwOnUnexpectedExitCode = false,
}) async {
  if (Platform.isWindows && !includeParentEnvironment) {
    const winEnvKeys = [
      'SYSTEMROOT',
      'TEMP',
      'TMP',
    ];
    environment = {
      for (final winEnvKey in winEnvKeys) winEnvKey: Platform.environment[winEnvKey]!,
      ...?environment,
    };
  }

  final printWorkingDir = workingDirectory != null && workingDirectory != Directory.current.uri;
  final commandString = [
    if (printWorkingDir) '(cd ${workingDirectory.toFilePath()};',
    ...?environment?.entries.map((entry) => '${entry.key}=${entry.value}'),
    executable,
    ...arguments.map((a) => a.contains(' ') ? "'$a'" : a),
    if (printWorkingDir) ')',
  ].join(' ');
  logger?.info('Running `$commandString`.');

  final stdoutBuffer = StringBuffer();
  final stderrBuffer = StringBuffer();
  final process = await Process.start(
    executable,
    arguments,
    workingDirectory: workingDirectory?.toFilePath(),
    environment: environment,
    includeParentEnvironment: includeParentEnvironment,
    runInShell: Platform.isWindows && !includeParentEnvironment,
  );

  final stdoutSub =
      process.stdout.transform(utf8.decoder).transform(const LineSplitter()).listen(captureOutput
          ? (s) {
              logger?.fine(s);
              stdoutBuffer.writeln(s);
            }
          : logger?.fine);
  final stderrSub =
      process.stderr.transform(utf8.decoder).transform(const LineSplitter()).listen(captureOutput
          ? (s) {
              logger?.severe(s);
              stderrBuffer.writeln(s);
            }
          : logger?.severe);

  final (exitCode, _, _) =
      await (process.exitCode, stdoutSub.asFuture<void>(), stderrSub.asFuture<void>()).wait;
  final result = RunProcessResult(
    pid: process.pid,
    command: commandString,
    exitCode: exitCode,
    stdout: stdoutBuffer.toString(),
    stderr: stderrBuffer.toString(),
  );
  if (throwOnUnexpectedExitCode && expectedExitCode != exitCode) {
    throw ProcessException(
      executable,
      arguments,
      "Full command string: '$commandString'.\n"
      "Exit code: '$exitCode'.\n"
      'For the output of the process check the logger output.',
    );
  }
  return result;
}

/// Drop in replacement of [ProcessResult].
class RunProcessResult {
  final int pid;

  final String command;

  final int exitCode;

  final String stderr;

  final String stdout;

  RunProcessResult({
    required this.pid,
    required this.command,
    required this.exitCode,
    required this.stderr,
    required this.stdout,
  });

  @override
  String toString() => '''command: $command
exitCode: $exitCode
stdout: $stdout
stderr: $stderr''';
}

class CMakeBuildException implements Exception {
  final String message;
  CMakeBuildException(this.message);

  @override
  String toString() => "CMakeBuildException(message=$message)";
}
