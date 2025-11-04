window.dash_clientside = Object.assign({}, window.dash_clientside, {
  ui: {
    spinCursor: function(loading_state) {
      const isLoading = loading_state && loading_state.is_loading;
      document.body.style.cursor = isLoading ? 'wait' : 'default';
      return window.dash_clientside.no_update;
    }
  }
});
