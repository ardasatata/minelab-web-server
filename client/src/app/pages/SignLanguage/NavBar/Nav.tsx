import * as React from 'react';
import styled from 'styled-components/macro';
import { ReactComponent as DocumentationIcon } from './assets/documentation-icon.svg';
import { ReactComponent as GithubIcon } from './assets/github-icon.svg';
import {
  ArrowUpOutlined,
  OrderedListOutlined,
  VideoCameraAddOutlined,
} from '@ant-design/icons';

export function Nav() {
  return (
    <Wrapper>
      <Item
        href={process.env.PUBLIC_URL + `/sign-language/list`}
        title="Documentation Page"
        rel="noopener noreferrer"
      >
        <OrderedListOutlined className={'mr-2'} />
        Video List
      </Item>
      <Item
        href={process.env.PUBLIC_URL + `/sign-language/upload`}
        title="Upload"
        rel="noopener noreferrer"
      >
        <ArrowUpOutlined className={'mr-2'} />
        Upload Video
      </Item>
      <Item
        href={process.env.PUBLIC_URL + `/sign-language/record`}
        title="Auto Record"
        rel="noopener noreferrer"
      >
        <VideoCameraAddOutlined className={'mr-2'} />
        Auto Record
      </Item>
    </Wrapper>
  );
}

const Wrapper = styled.nav`
  display: flex;
  margin-right: -1rem;
`;

const Item = styled.a`
  color: ${p => p.theme.primary};
  cursor: pointer;
  text-decoration: none;
  display: flex;
  padding: 0.25rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  align-items: center;

  &:hover {
    opacity: 0.8;
  }

  &:active {
    opacity: 0.4;
  }

  .icon {
    margin-right: 0.25rem;
  }
`;
