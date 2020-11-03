import React, { useState } from "react";
import { Button } from "../Button/Button";
import { Header } from "../Header/Header";
import { VideoJs } from "../Html5Video/Html5Video";
import { URLContext } from "../../contexts/URLcontext";
import Timer from "../Timer";
import "./pg2layout.css";

export const Pg2layout = () => {
  // anamoly_event = 0
  const [hidden, sethidden] = useState("none");
  const [timerstart, settimerstart] = useState(0);
  const [carasolehide, setcarasolehide] = useState("block");

  const hiddenStateChanger = () => {
    sethidden("block");
    settimerstart(1);
    setcarasolehide("none");
  };

  return (
    <URLContext.Consumer>
      {(context) => {
        const { isAnamoly } = context;
        return (
          <div className="maingrid">
            <Header />
            <div className="player">
              <VideoJs color={(isAnamoly === "false")?"green":"red"} />
            </div>

            <Button
              id="greenbutton"
              buttonStyle="btn--red-outline"
              buttonSize="btn--large"
              gridClass="normal"
            >
              Home
            </Button>
            <Button
              id="greenbutton"
              buttonStyle="btn--red-outline"
              buttonSize="btn--large"
              gridClass="anamoly"
            >
              Home
            </Button>

            <div className="info_">
              <div className="carasole" style={{ display: carasolehide }}>
                <Button
                  id="enbutton"
                  buttonStyle="btn--red-outline"
                  buttonSize="btn--large"
                  gridClass="rmal"
                  onClick={hiddenStateChanger}
                >
                  start
                </Button>
              </div>
              <div className="time" style={{ display: hidden }}>
                {timerstart ? (
                  <Timer initialMinute={timerstart} />
                ) : (
                  <div></div>
                )}
                <Button
                  id="greenbutton"
                  buttonStyle="btn--red-outline"
                  buttonSize="btn--large"
                  gridClass="timerstop"
                >
                  Stop Timer
                </Button>
              </div>
            </div>
          </div>
        );
      }}
    </URLContext.Consumer>
  );
};
