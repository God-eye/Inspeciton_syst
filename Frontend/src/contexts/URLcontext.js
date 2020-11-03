import React, { createContext, Component } from "react";

export const URLContext = createContext();

class URLContextProvider extends Component {
  state = {
    PythonURL: "",
    videoURL:"",
  };
  setPythonURL = (newURL) => {
    this.setState({ PythonURL: newURL });
  };
  render() {
    return (
      <URLContext.Provider
        value={{ ...this.state, setPythonURL: this.setPythonURL }}
      >
        {this.props.children}
      </URLContext.Provider>
    );
  }
}

export default URLContextProvider;
