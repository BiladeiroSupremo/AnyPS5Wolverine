# libSceNpCommerce technical debt

### Silent stubs

- [libSceNpCommerce](../../../core/libs/prx/libSceNpCommerce/Export.cpp) - the PS Store icon show/hide calls and `sceNpCommerceSetPsStoreIconLayout` do nothing

### Unknown function info

- [sceNpCommerceDialogOpen2](../../../core/libs/prx/libSceNpCommerce/Export.cpp) (libSceNpCommerce) - inferred signature follows `sceNpCommerceDialogOpen(const void*)`; accepts null or zero parameters and finishes as cancelled like the sibling call
