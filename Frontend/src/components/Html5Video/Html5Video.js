import React from "react";
import ReactPlayer from "react-player";
import test from "./test.mp4";
import "./Html5.css";

export const VideoJs = (color) => {
  console.log(color)
  return (
    <div className="player-wrapper" >
      <ReactPlayer
        className="react-player"
        url={test}
        width="100%"
        height="100%"
        style={{boxShadow:` 0 0 45px ${color.color}`}}
        controls={false}
        playing={true}
        muted = {true}
      />
    </div>
  );
};
