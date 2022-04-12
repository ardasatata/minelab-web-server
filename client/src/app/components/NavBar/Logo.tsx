import * as React from 'react';
import styled from 'styled-components/macro';
import ncuLogo from '../../../assets/ncuLogo.png';
import minelab from '../../../assets/minelab.png';

export function Logo() {
  return (
    <Wrapper>
      <img src={ncuLogo} className={'h-8 mr-2'}/>
      <img src={minelab} className={'h-8 mr-4'}/>
      <Title>Erhu Trainer | 二胡學</Title>
      <Description>Minelab 2022</Description>
    </Wrapper>
  );
}

const Wrapper = styled.div`
  display: flex;
  align-items: center;
`;

const Title = styled.div`
  font-size: 1.25rem;
  color: ${p => p.theme.text};
  font-weight: bold;
  margin-right: 1rem;
`;

const Description = styled.div`
  font-size: 0.875rem;
  color: ${p => p.theme.textSecondary};
  font-weight: normal;
`;
