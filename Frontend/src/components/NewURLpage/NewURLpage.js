import React, { Component } from "react";
import { URLContext } from "../../contexts/URLcontext";

export class NewURLpage extends Component {
  state = {
    value: null,
  };
  handlechange = (e) => {
    this.setState({ value: e.target.value });
  };

  render() {
    return (
      <URLContext.Consumer>
        {(context) => {
          const { PythonURL, setPythonURL } = context;
          const formSubmit = (a) => {
            a.preventDefault();
            setPythonURL(this.state.value);
            window.open(`${PythonURL}/setvid`, "_self");
          };
          return (
            <form onSubmit={formSubmit}>
              <label>
                Python server url:
                <input
                  type="text"
                  value={this.state.value}
                  placeholder="This is an examle URL. Get you own at :- <colabnbLink>"
                  onChange={this.handlechange}
                />{" "}
              </label>
              <input type="submit" value="SetURL" />
            </form>
          );
        }}
      </URLContext.Consumer>
    );
  }
}

export default NewURLpage;
