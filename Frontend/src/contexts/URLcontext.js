import React, { createContext, Component } from "react";

export const URLContext = createContext();

class URLContextProvider extends Component {
  state = {
    PythonURL: "http://0.0.0.0:5000",
    videoURL: "https://www.youtube.com/watch?v=3A7pQN5W08E",
    isAnamoly: false,
    videoName: "",
  };
  setPythonURL = (newURL) => {
    this.setState({ PythonURL: newURL });
  };
  setVideoURL = (vidURL) => {
    this.setState({ videoURL: vidURL });
  };
  setVideoName = (vidname) => {
    this.setState({ videoName: vidname });
  };
  setAnamoly = (isitAnamoly) => {
    this.setState({ isAnamoly: isitAnamoly });  
  };
  componentDidMount() {
    this.interval = setInterval(() => (
        fetch(`${this.state.PythonURL}/api/setanamoly`,
            {method: "GET"}).then((res)=>res.json()).then((data)=>(this.setAnamoly(data.anamoly)))), 1000)
    }
  componentWillUnmount() {
    clearInterval(this.interval);
  }
  render() {
    return (
      <URLContext.Provider
        value={{
          ...this.state,
          setPythonURL: this.setPythonURL,
          setVideoURL: this.setVideoURL,
          setVideoName: this.setVideoName,
        }}
      >
        {this.props.children}
      </URLContext.Provider>
    );
  }
}

export default URLContextProvider;
