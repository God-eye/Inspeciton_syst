import React,{useState} from "react";
import { Button } from "../Button/Button";
import { Header } from "../Header/Header";
import "./BtnGrp.css";

export const BtnGrp = () => {
  const [hiddenupload, sethiddenupload] = useState("none");
  const stateChanger =()=>{
      sethiddenupload("block")
  }

  return (
    <div className="temp">
      <Header />
      <div className="video_btn">
        <Button
          id="redbutton"
          buttonStyle="btn--red-outline"
          buttonSize="btn--large"
          gridClass="uploadshow"
          onClick = {stateChanger}
        >
          Home
        </Button>
        <Button
          id="greenbutton"
          buttonStyle="btn--green-outline"
          buttonSize="btn--large"
          gridClass="greenbutton1"
        >
          Home
        </Button>
      </div>
      <Button
        id="temp"
        buttonStyle="btn--red-outline"
        buttonSize="btn--large"
        gridClass="upload"
        hidebtn = {{display:hiddenupload}}
      >
        Home
      </Button>
    </div>
  );
};
