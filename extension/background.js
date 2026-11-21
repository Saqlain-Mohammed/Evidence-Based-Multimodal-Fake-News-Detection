const api = globalThis.browser || chrome;

api.runtime.onInstalled.addListener(() => {
  for (const context of ['selection', 'image', 'video']) {
    api.contextMenus.create({id: 'veritas-' + context, title: 'Examine with Veritas', contexts: [context]});
  }
});

api.contextMenus.onClicked.addListener(async info => {
  const mode = String(info.menuItemId).replace('veritas-', '');
  if (!['selection', 'image', 'video'].includes(mode)) return;
  const id = crypto.randomUUID();
  const content = mode === 'selection' ? {text: info.selectionText} : {mode, media: info.srcUrl};
  await api.storage.local.set({[id]: content});
  await api.tabs.create({url: api.runtime.getURL('popup.html') + '#' + id});
});
