import 'dart:io';

import 'package:logging/logging.dart';
import 'package:native_assets_cli/native_assets_cli.dart';

import 'native_toolchain_cmake.dart';

void main(List<String> args) async {
  await build(args, (config, output) async {
    final packageName = config.packageName;
    final builder = CMakeBuilder.library(
      name: packageName,
      assetName: '$packageName.dart',
      sourceDirectory: Directory("src/ggml").absolute,
      dartBuildFiles: ['hook/build.dart'],
      defines: {},
    );
    await builder.run(
      buildConfig: config,
      buildOutput: output,
      logger: Logger('')
        ..level = Level.ALL
        ..onRecord.listen((record) => print(record.message)),
    );
  });
}
