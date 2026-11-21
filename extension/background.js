const api = globalThis.browser || chrome;
api.runtime.onInstalled.addListener(()=>{api.contextMenus.create({id:'veritas-selection',title:'Examine with Veritas',contexts:['selection']});});
api.contextMenus.onClicked.addListener(async(info)=>{if(info.menuItemId==='veritas-selection'){await api.storage.local.set({pendingText:info.selectionText||''});await api.tabs.create({url:api.runtime.getURL('popup.html')});}});
