import React from "react";
import ReactPlayer from "react-player";
import { URLContext } from "../../contexts/URLcontext";
import "./Html5.css";

export const VideoJs = (color) => {
  console.log(color);
  return (
    <URLContext.Consumer>{(context) => {
      const { setPythonURL } = context;
      const formSubmit = (a) => {
        a.preventDefault();
        setPythonURL(this.state.value);
      };
      return (
      <div className="player-wrapper">
        <ReactPlayer
          className="react-player"
          url={test}
          width="100%"
          height="100%"
          style={{ boxShadow: ` 0 0 45px ${color.color}` }}
          controls={false}
          playing={true}
          muted={true}
        />
      </div>
      )}}
    </URLContext.Consumer>
  );
};
