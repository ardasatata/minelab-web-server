import * as React from 'react';
import styled from 'styled-components/macro';
import {
  ArrowUpOutlined,
  OrderedListOutlined,
  VideoCameraAddOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

export function Nav() {
  return (
    <Wrapper>
      <Item
        href={process.env.PUBLIC_URL + `/`}
        title="Auto Record"
        rel="noopener noreferrer"
      >
        <VideoCameraAddOutlined className={'mr-2'} />
        Auto Recording
      </Item>
      <Item
        href={process.env.PUBLIC_URL + `/list`}
        title="Documentation Page"
        rel="noopener noreferrer"
      >
        <OrderedListOutlined className={'mr-2'} />
        Analyzed Video
      </Item>
      <Item
        href={process.env.PUBLIC_URL + `/upload`}
        title="Upload"
        rel="noopener noreferrer"
      >
        <ArrowUpOutlined className={'mr-2'} />
        Upload Video
      </Item>
      <Item
        href={process.env.PUBLIC_URL + `/lesson`}
        title="Erhu Lessons"
        rel="noopener noreferrer"
      >
        <OrderedListOutlined className={'mr-2'} />
        Erhu Lessons
      </Item>
      <Item
        href={process.env.PUBLIC_URL + `/about`}
        title="Development Team"
        rel="noopener noreferrer"
      >
        <InfoCircleOutlined className={'mr-2'} />
        Devlopment Team
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
  padding: 0.25rem 0.50rem;
  font-size: 0.725rem;
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
