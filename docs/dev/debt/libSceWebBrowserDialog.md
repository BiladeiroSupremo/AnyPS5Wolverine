# libSceWebBrowserDialog technical debt

### Silent stubs

- [libSceWebBrowserDialog](../../../core/libs/prx/libSceWebBrowserDialog/Export.cpp) - `sceWebBrowserDialogOpen` and `sceWebBrowserDialogOpenForPredeterminedContent` run the common-dialog state machine without displaying an interactive browser, and `sceWebBrowserDialogGetResult` reports that the user cancelled. Minecraft Bedrock (PPSA17221) imports `sceWebBrowserDialogOpenForPredeterminedContent`
