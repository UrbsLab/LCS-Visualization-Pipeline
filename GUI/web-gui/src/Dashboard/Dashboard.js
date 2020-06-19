import React from "react";
import SignInPage from "./SignInPage";
import DashboardPage from "./DashboardPage";

class Dashboard extends React.Component {
  constructor() {
    super();
    this.state = {
      hasLoggedIn: true,
    };
    this.logout = this.logout.bind(this);
    this.login = this.login.bind(this);
  }

  logout() {
    this.setState({ hasLoggedIn: false });
  }

  login() {
    this.setState({ hasLoggedIn: true });
  }
  render() {
    return (
      <div>
        {this.state.hasLoggedIn ? (
          <DashboardPage logoutfunc={this.logout} />
        ) : (
          <SignInPage loginfunc={this.login} />
        )}
      </div>
    );
  }
}

export default Dashboard;
