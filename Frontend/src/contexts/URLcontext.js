import React, { createContext, Component } from "react";

export const URLContext = createContext();

class URLContextProvider extends Component {
  state = {
    PythonURL: "",
    videoURL:"https://www.youtube.com/watch?v=3A7pQN5W08E",
    isAnamoly:false,
    videoName:``,
  };
  setPythonURL = (newURL) => {
    this.setState({ PythonURL: newURL });
  };
  setVideoURL = (vidURL) => {
      this.setState({videoURL: vidURL})
  }
  render() {
    return (
      <URLContext.Provider
        value={{ ...this.state, setPythonURL: this.setPythonURL, setVideoURL: this.setVideoURL }}
      >
        {this.props.children}
      </URLContext.Provider>
    );
  }
}

export default URLContextProvider;
