const fs=require('node:fs/promises');
const path=require('node:path');
const {withAndroidManifest,withDangerousMod}=require('expo/config-plugins');

module.exports=function localNetworking(config){
  config=withAndroidManifest(config,mod=>{
    mod.modResults.manifest.application[0].$['android:networkSecurityConfig']='@xml/veritas_network_security';
    return mod;
  });
  return withDangerousMod(config,['android',async mod=>{
    const directory=path.join(mod.modRequest.platformProjectRoot,'app/src/main/res/xml');
    await fs.mkdir(directory,{recursive:true});
    await fs.writeFile(path.join(directory,'veritas_network_security.xml'),
      '<network-security-config><base-config cleartextTrafficPermitted="false"/><domain-config cleartextTrafficPermitted="true"><domain>10.0.2.2</domain><domain>localhost</domain><domain>127.0.0.1</domain></domain-config></network-security-config>\n');
    return mod;
  }]);
};
